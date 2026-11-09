package server

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func uploadFixture(t *testing.T, registry string) (model.Name, manifest.Layer, string) {
	t.Helper()
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	data := []byte("small model blob")
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	path, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	u, _ := url.Parse(registry)
	n := model.Name{ProtocolScheme: "http", Host: u.Host, Namespace: "library", Model: "upload"}
	return n, manifest.Layer{Digest: digest, Size: int64(len(data))}, string(data)
}

func uploadKey(n model.Name, digest string) string {
	return n.BaseURL().JoinPath("v2", n.DisplayNamespaceModel(), "blobs", digest).String()
}

func waitUploadReferences(t *testing.T, key string, count int32) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if value, ok := blobUploadManager.Load(key); ok && value.(*blobUpload).references.Load() == count {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("upload did not acquire %d consumers", count)
}

func waitUploadGone(t *testing.T, key string) {
	t.Helper()
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if _, ok := blobUploadManager.Load(key); !ok {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("cancelled upload kept its background worker")
}

func TestSharedUploadPreparationFailureReachesEveryConsumer(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		close(started)
		select {
		case <-release:
			w.WriteHeader(http.StatusNotFound)
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(registry.Close)
	n, layer, _ := uploadFixture(t, registry.URL)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	results := make(chan error, 2)
	run := func() {
		results <- uploadBlob(ctx, n, layer, &registryOptions{Insecure: true}, func(api.ProgressResponse) {})
	}
	go run()
	<-started
	go run()
	waitUploadReferences(t, uploadKey(n, layer.Digest), 2)
	once.Do(func() { close(release) })
	for range 2 {
		if err := <-results; !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("preparation error = %v", err)
		}
	}
}

func TestSharedUploadCancellationPreservesOtherConsumer(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	var posts atomic.Int32
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		posts.Add(1)
		close(started)
		select {
		case <-release:
			w.WriteHeader(http.StatusCreated)
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(registry.Close)
	n, layer, _ := uploadFixture(t, registry.URL)
	first, cancel := context.WithCancel(t.Context())
	second, stop := context.WithTimeout(t.Context(), 3*time.Second)
	defer stop()
	results := make(chan error, 2)
	go func() {
		results <- uploadBlob(first, n, layer, &registryOptions{Insecure: true}, func(api.ProgressResponse) {})
	}()
	<-started
	go func() {
		results <- uploadBlob(second, n, layer, &registryOptions{Insecure: true}, func(api.ProgressResponse) {})
	}()
	waitUploadReferences(t, uploadKey(n, layer.Digest), 2)
	cancel()
	if err := <-results; !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled consumer: %v", err)
	}
	once.Do(func() { close(release) })
	if err := <-results; err != nil {
		t.Fatalf("remaining consumer: %v", err)
	}
	if posts.Load() != 1 {
		t.Fatalf("sessions=%d, want 1", posts.Load())
	}
}

func TestSharedUploadLastConsumerCancelsPreparationAndCanRetry(t *testing.T) {
	started, cancelled := make(chan struct{}), make(chan struct{})
	var posts atomic.Int32
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if posts.Add(1) == 1 {
			close(started)
			<-r.Context().Done()
			close(cancelled)
			return
		}
		w.WriteHeader(http.StatusCreated)
	}))
	t.Cleanup(registry.Close)
	n, layer, _ := uploadFixture(t, registry.URL)
	ctx, cancel := context.WithCancel(t.Context())
	result := make(chan error, 1)
	go func() {
		result <- uploadBlob(ctx, n, layer, &registryOptions{Insecure: true}, func(api.ProgressResponse) {})
	}()
	<-started
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("preparation was not cancelled")
	}
	ctx, stop := context.WithTimeout(t.Context(), 3*time.Second)
	defer stop()
	if err := uploadBlob(ctx, n, layer, &registryOptions{Insecure: true}, func(api.ProgressResponse) {}); err != nil {
		t.Fatal(err)
	}
	if posts.Load() != 2 {
		t.Fatalf("sessions=%d, want 2", posts.Load())
	}
}

func TestSharedUploadTargetsAreIndependent(t *testing.T) {
	started, release := make(chan string, 2), make(chan struct{})
	var once sync.Once
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		started <- r.URL.Path
		select {
		case <-release:
			w.WriteHeader(http.StatusCreated)
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(registry.Close)
	n, layer, _ := uploadFixture(t, registry.URL)
	other := n
	other.Model = "different"
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	results := make(chan error, 2)
	for _, target := range []model.Name{n, other} {
		go func() {
			results <- uploadBlob(ctx, target, layer, &registryOptions{Insecure: true}, func(api.ProgressResponse) {})
		}()
	}
	paths := map[string]bool{}
	for range 2 {
		select {
		case path := <-started:
			paths[path] = true
		case <-ctx.Done():
			t.Fatal("different targets shared the same upload")
		}
	}
	if len(paths) != 2 {
		t.Fatalf("targets = %v", paths)
	}
	once.Do(func() { close(release) })
	for range 2 {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
}

func TestBlobUploadRelativeLocationsPreserveBytesAndCommit(t *testing.T) {
	var uploaded atomic.Value
	var committed atomic.Bool
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPost:
			w.Header().Set("Location", "/upload/session")
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPatch:
			data, _ := io.ReadAll(r.Body)
			uploaded.Store(string(data))
			w.Header().Set("Location", "/upload/session")
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPut:
			if r.URL.Path != "/upload/session" || !strings.HasPrefix(r.URL.Query().Get("digest"), "sha256:") {
				t.Error("invalid commit URL")
			}
			committed.Store(true)
			w.WriteHeader(http.StatusCreated)
		}
	}))
	t.Cleanup(registry.Close)
	n, layer, body := uploadFixture(t, registry.URL)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	if err := uploadBlob(ctx, n, layer, &registryOptions{Insecure: true}, func(api.ProgressResponse) {}); err != nil {
		t.Fatal(err)
	}
	if uploaded.Load() != body || !committed.Load() {
		t.Fatalf("uploaded=%q committed=%v", uploaded.Load(), committed.Load())
	}
}

func TestBlobUploadMountedBlobPublishesFinalProgress(t *testing.T) {
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.Method != http.MethodPost {
			t.Errorf("mounted blob unexpectedly sent %s", r.Method)
		}
		w.WriteHeader(http.StatusCreated)
	}))
	t.Cleanup(registry.Close)
	n, layer, _ := uploadFixture(t, registry.URL)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	var latest api.ProgressResponse
	if err := uploadBlob(ctx, n, layer, &registryOptions{Insecure: true}, func(value api.ProgressResponse) {
		latest = value
	}); err != nil {
		t.Fatal(err)
	}
	if latest.Digest != layer.Digest || latest.Total != layer.Size || latest.Completed != layer.Size {
		t.Fatalf("mounted blob lost completion progress: %+v", latest)
	}
}

func TestBlobUploadCancellationInterruptsRetryBackoff(t *testing.T) {
	failed := make(chan struct{})
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case http.MethodPost:
			w.Header().Set("Location", "/upload/session")
			w.WriteHeader(http.StatusAccepted)
		case http.MethodPatch:
			close(failed)
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	t.Cleanup(registry.Close)
	n, layer, _ := uploadFixture(t, registry.URL)
	ctx, cancel := context.WithCancel(t.Context())
	result := make(chan error, 1)
	go func() {
		result <- uploadBlob(ctx, n, layer, &registryOptions{Insecure: true}, func(api.ProgressResponse) {})
	}()
	<-failed
	cancel()
	if err := <-result; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	waitUploadGone(t, uploadKey(n, layer.Digest))
}
