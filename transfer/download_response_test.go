package transfer

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestValidateDownloadResponse(t *testing.T) {
	for _, tc := range []struct {
		name, contentRange  string
		status              int
		offset, size, total int64
		length              int64
		invalid             bool
	}{
		{name: "full", status: 200, size: 10, total: 10, length: 10},
		{name: "chunked full", status: 200, size: 10, total: 10, length: -1},
		{name: "empty blob", status: 200},
		{name: "nonempty declared empty", status: 200, length: 7, invalid: true},
		{name: "negative expected size", status: 200, size: -1, total: -1, length: 7, invalid: true},
		{name: "partial", status: 206, offset: 4, size: 3, total: 10, length: 3, contentRange: "bytes 4-6/10"},
		{name: "chunked partial", status: 206, offset: 4, size: 3, total: 10, length: -1, contentRange: "bytes 4-6/10"},
		{name: "unknown representation size", status: 206, offset: 4, size: 3, total: 10, length: 3, contentRange: "bytes 4-6/*"},
		{name: "case insensitive unit", status: 206, offset: 4, size: 3, total: 10, length: 3, contentRange: "BYTES 4-6/10"},
		{name: "ignored resume range", status: 200, offset: 4, size: 6, total: 10, length: 10, invalid: true},
		{name: "ignored multipart range", status: 200, size: 3, total: 10, length: 10, invalid: true},
		{name: "missing range", status: 206, offset: 4, size: 3, total: 10, length: 3, invalid: true},
		{name: "wrong first byte", status: 206, offset: 4, size: 3, total: 10, length: 3, contentRange: "bytes 0-2/10", invalid: true},
		{name: "wrong last byte", status: 206, offset: 4, size: 3, total: 10, length: 3, contentRange: "bytes 4-7/10", invalid: true},
		{name: "changed representation", status: 206, offset: 4, size: 3, total: 10, length: 3, contentRange: "bytes 4-6/11", invalid: true},
		{name: "wrong length", status: 206, offset: 4, size: 3, total: 10, length: 2, contentRange: "bytes 4-6/10", invalid: true},
		{name: "wrong unit", status: 206, offset: 4, size: 3, total: 10, length: 3, contentRange: "items 4-6/10", invalid: true},
		{name: "malformed", status: 206, offset: 4, size: 3, total: 10, length: 3, contentRange: "bytes 4-6/10/11", invalid: true},
		{name: "negative", status: 206, offset: 4, size: 3, total: 10, length: 3, contentRange: "bytes -4-6/10", invalid: true},
		{name: "overflow", status: 206, offset: 4, size: 3, total: 10, length: 3, contentRange: "bytes 4-18446744073709551615/10", invalid: true},
		{name: "expected range overflow", status: 206, offset: math.MaxInt64, size: 2, total: math.MaxInt64, length: 2, invalid: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := &unreadTransferBody{}
			response := &http.Response{StatusCode: tc.status, ContentLength: tc.length, Header: http.Header{"Content-Range": []string{tc.contentRange}}, Body: body}
			err := ValidateDownloadResponse(response, tc.offset, tc.size, tc.total)
			if (err != nil) != tc.invalid || (tc.invalid && !errors.Is(err, ErrInvalidDownloadRange)) {
				t.Fatalf("response validation error = %v, invalid = %v", err, tc.invalid)
			}
			if body.reads != 0 {
				t.Fatal("validation consumed unverified blob bytes")
			}
		})
	}
}

func TestTransferRejectsWrongResumedRangeWithoutReadingOrRetry(t *testing.T) {
	directory := t.TempDir()
	digest := "sha256:" + strings.Repeat("a", 64)
	total := int64(resumeThreshold)
	partial := filepath.Join(directory, digestToPath(digest)+".tmp")
	file, err := os.Create(partial)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Truncate(3); err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteAt([]byte("old"), 0); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	before, err := os.Stat(partial)
	if err != nil {
		t.Fatal(err)
	}
	body := &unreadTransferBody{}
	requests := 0
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		requests++
		if requests == 1 { // resolve only inspects the successful response headers
			return &http.Response{StatusCode: 200, Body: io.NopCloser(bytes.NewReader(nil)), Header: make(http.Header)}, nil
		}
		if got := req.Header.Get("Range"); got != "bytes=3-" {
			t.Errorf("resume request = %q", got)
		}
		return &http.Response{StatusCode: 206, Body: body, ContentLength: total - 3, Header: http.Header{"Content-Range": []string{fmt.Sprintf("bytes 0-%d/%d", total-4, total)}}}, nil
	})}
	downloader := &downloader{client: client, baseURL: "https://fixture.invalid", destDir: directory, repository: "library/test", progress: newProgressTracker(total, func(int64, int64) {})}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	err = downloader.download(ctx, Blob{Digest: digest, Size: total})
	if !errors.Is(err, ErrInvalidDownloadRange) || requests != 2 || body.reads != 0 || !body.closed {
		t.Fatalf("invalid resume: error=%v requests=%d reads=%d closed=%v", err, requests, body.reads, body.closed)
	}
	after, err := os.Stat(partial)
	if err != nil {
		t.Fatal(err)
	}
	contents, err := os.ReadFile(partial)
	if err != nil || string(contents) != "old" || after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) || downloader.progress.completed.Load() != 0 {
		t.Fatalf("valid partial state changed: contents=%q size=%d error=%v", contents, after.Size(), err)
	}
}

func TestTransferPreservesEmptyAndChunkedDownloads(t *testing.T) {
	for _, data := range [][]byte{nil, []byte("chunked")} {
		t.Run(fmt.Sprintf("size%d", len(data)), func(t *testing.T) {
			digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				w.WriteHeader(200)
				w.(http.Flusher).Flush() // unknown HTTP Content-Length, known blob size
				_, _ = w.Write(data)
			}))
			defer server.Close()
			directory := t.TempDir()
			if err := Download(t.Context(), DownloadOptions{Blobs: []Blob{{Digest: digest, Size: int64(len(data))}}, BaseURL: server.URL, Client: server.Client(), DestDir: directory}); err != nil {
				t.Fatal(err)
			}
			got, err := os.ReadFile(filepath.Join(directory, digestToPath(digest)))
			if err != nil || !bytes.Equal(got, data) {
				t.Fatalf("chunked contents = %q, error=%v", got, err)
			}
		})
	}
}
