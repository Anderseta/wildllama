package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/transfer"
)

func TestDownloadChunkRejectsWrongBytesBeforeWriting(t *testing.T) {
	for _, tc := range []struct {
		name, span, body string
		status           int
	}{
		{"ignored range", "", "wrong", 200},
		{"wrong range", "bytes 0-2/12", "bad", 206},
		{"changed total", "bytes 6-8/13", "bad", 206},
		{"missing range", "", "bad", 206},
		{"not found", "", "bad", 404},
		{"server error", "", "bad", 503},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if req.Header.Get("Range") != "bytes=6-8" {
					t.Errorf("Range = %q", req.Header.Get("Range"))
				}
				w.Header().Set("Content-Range", tc.span)
				w.WriteHeader(tc.status)
				fmt.Fprint(w, tc.body)
			}))
			defer server.Close()
			address, _ := url.Parse(server.URL)
			download := &blobDownload{Name: filepath.Join(t.TempDir(), "blob"), Total: 12}
			download.Completed.Store(2)
			part := &blobDownloadPart{Offset: 4, Size: 5, blobDownload: download}
			part.Completed.Store(2)
			var written bytes.Buffer
			err := download.downloadChunk(t.Context(), server.Client(), address, &written, part)
			if err == nil || written.Len() != 0 || part.Completed.Load() != 2 || download.Completed.Load() != 2 {
				t.Fatalf("bad response advanced valid data: bytes=%q part=%d total=%d err=%v", written.String(), part.Completed.Load(), download.Completed.Load(), err)
			}
			if _, err := os.Stat(part.Name()); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("invalid response published resume metadata")
			}
		})
	}
}

func TestDownloadPartFailureCancelsSiblingBackoff(t *testing.T) {
	t.Setenv("WILDLLAMA_MIN_FREE_DISK_MIB", "0")
	firstFailed := make(chan struct{})
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.Header.Get("Range") == "bytes=0-0" {
			w.WriteHeader(503)
			close(firstFailed)
			return
		}
		select {
		case <-firstFailed:
		case <-req.Context().Done():
			return
		}
		// The sibling has entered its one-second retry backoff by this point.
		time.Sleep(100 * time.Millisecond)
		w.Header().Set("Content-Range", "bytes 0-0/2")
		w.WriteHeader(206)
		fmt.Fprint(w, "x")
	}))
	defer cdn.Close()
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		http.Redirect(w, req, cdn.URL, http.StatusTemporaryRedirect)
	}))
	defer registry.Close()
	download := &blobDownload{Name: filepath.Join(t.TempDir(), "blob"), Total: 2, Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"}
	for i := range 2 {
		if err := download.newPart(int64(i), 1); err != nil {
			t.Fatal(err)
		}
	}
	address, _ := url.Parse(registry.URL)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	start := time.Now()
	err := download.run(ctx, address, &registryOptions{Insecure: true})
	if !errors.Is(err, transfer.ErrInvalidDownloadRange) || time.Since(start) >= 700*time.Millisecond {
		t.Fatalf("fatal part error waited for sibling backoff: %v after %v", err, time.Since(start))
	}
}
