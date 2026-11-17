package transfer

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func TestDownloadPlanReservesUniqueMissingBytes(t *testing.T) {
	dir := t.TempDir()
	a := Blob{Digest: fmt.Sprintf("sha256:%x", sha256.Sum256([]byte("partial"))), Size: resumeThreshold}
	b := Blob{Digest: fmt.Sprintf("sha256:%x", sha256.Sum256([]byte("stored"))), Size: 6}
	for file, data := range map[string]string{digestToPath(a.Digest) + ".tmp": "abc", digestToPath(b.Digest): "stored"} {
		if err := os.WriteFile(filepath.Join(dir, file), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	stop := errors.New("disk budget reached")
	opts := DownloadOptions{Blobs: []Blob{a, b, a, b}, DestDir: dir, CheckSpace: func(_ context.Context, path string, n int64) error {
		if path != dir || n != a.Size-3 {
			t.Fatalf("unexpected disk plan: %s %d", path, n)
		}
		return stop
	}}
	if err := Download(t.Context(), opts); !errors.Is(err, stop) {
		t.Fatalf("preflight: %v", err)
	}
	c := a
	c.Size++
	opts.Blobs = []Blob{a, c}
	if err := Download(t.Context(), opts); err == nil || errors.Is(err, stop) {
		t.Fatalf("conflicting sizes: %v", err)
	}
}

func TestRangeResumePreservesPrefixOnFailure(t *testing.T) {
	denied := errors.New("no space")
	for _, tc := range []struct {
		name, data   string
		cancel, deny bool
		want         error
	}{
		{"short prefix", "a", false, false, io.EOF},
		{"disk reservation", "abc", false, true, denied},
		{"cancel verification", "abc", true, false, context.Canceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d := &downloader{destDir: t.TempDir(), progress: newProgressTracker(6, func(int64, int64) {})}
			b := Blob{Digest: fmt.Sprintf("sha256:%x", sha256.Sum256([]byte("abcdef"))), Size: 6}
			tmp := filepath.Join(d.destDir, digestToPath(b.Digest)+".tmp")
			if err := os.WriteFile(tmp, []byte(tc.data), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(nil)
			if tc.cancel {
				cancel(context.Canceled)
			}
			if tc.deny {
				d.reserveSpace = func(context.Context, string, int64) (DiskReservation, error) { return nil, denied }
			}
			body := bytes.NewReader([]byte("def"))
			n, err := d.save(ctx, cancel, b, body, 3)
			data, readErr := os.ReadFile(tmp)
			if !errors.Is(err, tc.want) || n != 0 || d.progress.completed.Load() != 0 || body.Len() != 3 || readErr != nil || string(data) != tc.data {
				t.Fatalf("resume: err=%v n=%d progress=%d remaining=%d prefix=%q read error=%v", err, n, d.progress.completed.Load(), body.Len(), data, readErr)
			}
		})
	}
}

func TestTransferNegativeConcurrencyAndDuplicateDownloads(t *testing.T) {
	data := []byte("only one transfer")
	blob := Blob{Digest: fmt.Sprintf("sha256:%x", sha256.Sum256(data)), Size: int64(len(data))}
	var checks, files, mounts atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			files.Add(1)
			w.Write(data)
		case http.MethodHead:
			checks.Add(1)
			w.WriteHeader(404)
		case http.MethodPost:
			mounts.Add(1)
			w.WriteHeader(201)
		default:
			t.Errorf("unexpected request %s", r.Method)
		}
	}))
	defer server.Close()
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	dir := t.TempDir()
	var end int64
	opts := DownloadOptions{Blobs: []Blob{blob, blob}, BaseURL: server.URL, Client: server.Client(), DestDir: dir, Concurrency: -1, Progress: func(done, total int64) {
		if total != blob.Size || done > total {
			t.Errorf("duplicate progress: %d/%d", done, total)
		}
		end = done
	}}
	if err := Download(ctx, opts); err != nil {
		t.Fatal(err)
	}
	if files.Load() != 2 || end != blob.Size {
		t.Fatalf("requests including discovery=%d progress=%d", files.Load(), end)
	}
	end = 0
	if err := Download(ctx, opts); err != nil || end != blob.Size || files.Load() != 2 {
		t.Fatalf("cached progress=%d err=%v", end, err)
	}
	if err := Upload(ctx, UploadOptions{Blobs: []Blob{blob}, BaseURL: server.URL, Client: server.Client(), SrcDir: dir, Concurrency: -1}); err != nil {
		t.Fatal(err)
	}
	if checks.Load() != 1 || mounts.Load() != 1 {
		t.Fatalf("upload checks=%d mounts=%d", checks.Load(), mounts.Load())
	}
}
