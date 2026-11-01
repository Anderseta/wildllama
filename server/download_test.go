package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/transfer"
	"github.com/ollama/ollama/types/model"
)

func TestDownloadChunkRedirects(t *testing.T) {
	for _, tc := range []struct {
		name     string
		location string
		insecure bool
		wantErr  string
		wantHits int32
	}{
		{name: "public", location: "https://8.8.8.8/data", wantHits: 2},
		{name: "private", location: "https://127.0.0.1/data", wantErr: "not allowed", wantHits: 1},
		{name: "downgrade", location: "http://8.8.8.8/data", wantErr: "not allowed", wantHits: 1},
		{name: "lookup", wantErr: "not allowed"},
		{name: "insecure", insecure: true, wantHits: 2},
		{name: "loop", location: "https://8.8.4.4/start", wantErr: "stopped after 10 redirects", wantHits: 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var hits atomic.Int32
			var location string
			server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				if r.URL.Path == "/start" {
					http.Redirect(w, r, location, http.StatusTemporaryRedirect)
					return
				}
				w.Write([]byte("x"))
			}))
			t.Cleanup(server.Close)
			location = tc.location
			if location == "" {
				location = server.URL + "/data"
			}
			client := transfer.NewRedirectClient("https://registry.example", tc.insecure)
			t.Cleanup(client.CloseIdleConnections)
			tr := server.Client().Transport.(*http.Transport).Clone()
			client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
				if err := transfer.ValidateRedirectTarget(req.Context(), req.URL, "https://registry.example", tc.insecure); err != nil {
					return nil, err
				}
				return tr.RoundTrip(req)
			})
			tr.TLSClientConfig.ServerName = "127.0.0.1"
			dial := tr.DialContext
			tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
				// Route only the public fixture addresses to the test server.
				if addr == "8.8.4.4:443" || addr == "8.8.8.8:443" || addr == "8.8.8.8:80" {
					return new(net.Dialer).DialContext(ctx, network, server.Listener.Addr().String())
				}
				return dial(ctx, network, addr)
			}
			requestURL, _ := url.Parse("https://8.8.4.4/start")
			if tc.name == "lookup" {
				requestURL, _ = url.Parse(strings.Replace(server.URL, "127.0.0.1", "localhost", 1) + "/data")
			}
			download := &blobDownload{Name: filepath.Join(t.TempDir(), "blob")}
			part := &blobDownloadPart{Size: 1, blobDownload: download}
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			var body bytes.Buffer
			err := download.downloadChunk(ctx, client, requestURL, &body, part)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("downloadChunk() error = %v, want %q", err, tc.wantErr)
				}
			} else if err != nil || body.String() != "x" {
				t.Fatalf("downloadChunk() = %q, %v, want x, nil", body.String(), err)
			}
			if got := hits.Load(); got != tc.wantHits {
				t.Errorf("requests = %d, want %d", got, tc.wantHits)
			}
		})
	}
}

func TestSharedBlobPreparationFailureReachesEveryWaiter(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	started, release := make(chan struct{}), make(chan struct{})
	var heads atomic.Int32
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodHead {
			t.Errorf("unexpected request: %s", r.Method)
			return
		}
		if heads.Add(1) == 1 {
			close(started)
		}
		select {
		case <-release:
			w.WriteHeader(http.StatusNotFound)
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(registry.Close)
	opts := sharedDownloadOptions(registry.URL)
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	results := make(chan error, 2)
	go func() { _, err := downloadBlob(ctx, opts); results <- err }()
	<-started
	go func() { _, err := downloadBlob(ctx, opts); results <- err }()
	waitForDownloadReferences(t, opts.digest, 2)
	close(release)
	for range 2 {
		if err := <-results; !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("prepare error = %v", err)
		}
	}
	if heads.Load() != 1 {
		t.Fatalf("HEAD requests = %d, want 1", heads.Load())
	}
}

func TestSharedBlobCancellationKeepsRemainingConsumer(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	started, release, headCancelled := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var gets atomic.Int32
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gets.Add(1)
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte("x"))
	}))
	t.Cleanup(cdn.Close)
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			close(started)
			select {
			case <-release:
				w.Header().Set("Content-Length", "1")
			case <-r.Context().Done():
				close(headCancelled)
			}
			return
		}
		gets.Add(1)
		http.Redirect(w, r, cdn.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(registry.Close)
	opts := sharedDownloadOptions(registry.URL)
	first, cancel := context.WithCancel(t.Context())
	second, stop := context.WithTimeout(t.Context(), 3*time.Second)
	defer stop()
	firstResult, secondResult := make(chan error, 1), make(chan error, 1)
	go func() { _, err := downloadBlob(first, opts); firstResult <- err }()
	<-started
	go func() { _, err := downloadBlob(second, opts); secondResult <- err }()
	waitForDownloadReferences(t, opts.digest, 2)
	cancel()
	if err := <-firstResult; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	select {
	case <-headCancelled:
		t.Fatal("remaining consumer lost its preparation")
	default:
	}
	close(release)
	if err := <-secondResult; err != nil {
		t.Fatal(err)
	}
	if gets.Load() != 2 {
		t.Fatalf("GET requests = %d, want discovery and one part", gets.Load())
	}
}

func TestSharedBlobAllConsumersCancelPreparationWithoutOrphanTransfer(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	started, headCancelled := make(chan struct{}), make(chan struct{})
	var gets atomic.Int32
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodHead {
			gets.Add(1)
			return
		}
		close(started)
		<-r.Context().Done()
		close(headCancelled)
	}))
	t.Cleanup(registry.Close)
	opts := sharedDownloadOptions(registry.URL)
	ctx, cancel := context.WithCancel(t.Context())
	results := make(chan error, 2)
	go func() { _, err := downloadBlob(ctx, opts); results <- err }()
	<-started
	go func() { _, err := downloadBlob(ctx, opts); results <- err }()
	waitForDownloadReferences(t, opts.digest, 2)
	cancel()
	for range 2 {
		if err := <-results; !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	}
	select {
	case <-headCancelled:
	case <-time.After(time.Second):
		t.Fatal("orphan HEAD request")
	}
	deadline := time.Now().Add(time.Second)
	for {
		if _, ok := blobDownloadManager.Load(opts.digest); !ok {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("orphan download entry")
		}
		time.Sleep(time.Millisecond)
	}
	if gets.Load() != 0 {
		t.Fatal("GET started after all consumers cancelled")
	}
}

func sharedDownloadOptions(registry string) downloadOpts {
	u, _ := url.Parse(registry)
	return downloadOpts{n: model.Name{Host: u.Host, ProtocolScheme: "http", Namespace: "library", Model: "shared"},
		digest:  "sha256:0000000000000000000000000000000000000000000000000000000000000000",
		regOpts: &registryOptions{Insecure: true}, fn: func(api.ProgressResponse) {}}
}

func waitForDownloadReferences(t *testing.T, digest string, count int32) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if value, ok := blobDownloadManager.Load(digest); ok && value.(*blobDownload).references.Load() == count {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("download consumers did not join")
}

type observedCancelledContext struct {
	context.Context
	checked chan struct{}
	once    sync.Once
}

func (c *observedCancelledContext) Err() error {
	c.once.Do(func() { close(c.checked) })
	return c.Context.Err()
}

func TestSharedBlobFinishingTransferIsNotDownloadedAgain(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	var requests atomic.Int32
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { requests.Add(1); w.WriteHeader(500) }))
	t.Cleanup(registry.Close)
	opts := sharedDownloadOptions(registry.URL)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	closing := &observedCancelledContext{Context: ctx, checked: make(chan struct{})}
	previous := &blobDownload{context: closing, CancelFunc: func() {}, done: make(chan struct{})}
	blobDownloadManager.Store(opts.digest, previous)
	t.Cleanup(func() { blobDownloadManager.CompareAndDelete(opts.digest, previous) })
	result := make(chan error, 1)
	go func() { _, err := downloadBlob(t.Context(), opts); result <- err }()
	<-closing.checked
	filename, err := manifest.BlobsPath(opts.digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filename, []byte("finished"), 0644); err != nil {
		t.Fatal(err)
	}
	blobDownloadManager.CompareAndDelete(opts.digest, previous)
	close(previous.done)
	if err := <-result; err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 0 {
		t.Fatal("completed blob unnecessarily downloaded again")
	}
}

func TestSharedBlobCannotJoinWhileLastConsumerCancels(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	started, proceed := make(chan struct{}), make(chan struct{})
	download := &blobDownload{context: ctx, CancelFunc: func() { close(started); <-proceed; cancel() }}
	if !download.acquire() {
		t.Fatal("initial consumer rejected")
	}
	released := make(chan struct{})
	go func() { download.release(); close(released) }()
	<-started
	joined := make(chan bool, 1)
	go func() { joined <- download.acquire() }()
	select {
	case accepted := <-joined:
		close(proceed)
		t.Fatalf("joined cancelling transfer: %v", accepted)
	case <-time.After(20 * time.Millisecond):
	}
	close(proceed)
	<-released
	if <-joined {
		t.Fatal("cancelled transfer accepted a new consumer")
	}
}

func TestDownloadDiscoveryRedirectOrigin(t *testing.T) {
	t.Run("same origin", func(t *testing.T) {
		var hits atomic.Int32
		registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits.Add(1)
			if r.URL.Path == "/start" {
				http.Redirect(w, r, "/data", http.StatusTemporaryRedirect)
				return
			}
			fmt.Fprint(w, "ok")
		}))
		defer registry.Close()
		base, _ := url.Parse(registry.URL)
		client := &http.Client{CheckRedirect: downloadDiscoveryRedirect(base)}
		resp, err := client.Get(registry.URL + "/start")
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK || hits.Load() != 2 {
			t.Fatalf("same-origin redirect: status %d, hits %d", resp.StatusCode, hits.Load())
		}
	})

	t.Run("different port", func(t *testing.T) {
		var targetHits atomic.Int32
		target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			targetHits.Add(1)
		}))
		defer target.Close()
		registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, target.URL+"/private", http.StatusFound)
		}))
		defer registry.Close()
		base, _ := url.Parse(registry.URL)
		client := &http.Client{CheckRedirect: downloadDiscoveryRedirect(base)}
		resp, err := client.Get(registry.URL + "/start")
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusFound || targetHits.Load() != 0 {
			t.Fatalf("different-port redirect: status %d, target hits %d", resp.StatusCode, targetHits.Load())
		}
	})

	t.Run("https downgrade", func(t *testing.T) {
		var registryURL string
		registry := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			loc := strings.Replace(registryURL, "https://", "http://", 1) + "/data"
			http.Redirect(w, r, loc, http.StatusTemporaryRedirect)
		}))
		defer registry.Close()
		registryURL = registry.URL
		base, _ := url.Parse(registry.URL)
		client := registry.Client()
		client.CheckRedirect = downloadDiscoveryRedirect(base)
		resp, err := client.Get(registry.URL + "/start")
		if err == nil {
			resp.Body.Close()
		}
		if err == nil || !strings.Contains(err.Error(), "not allowed") {
			t.Fatalf("HTTPS downgrade was followed: %v", err)
		}
	})
}

func TestDownloadRedirect(t *testing.T) {
	var hits atomic.Int32
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Write([]byte("x"))
	}))
	t.Cleanup(cdn.Close)
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, cdn.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(registry.Close)
	requestURL, _ := url.Parse(strings.Replace(registry.URL, "127.0.0.1", "localhost", 1))
	for _, insecure := range []bool{false, true} {
		download := &blobDownload{
			Name: filepath.Join(t.TempDir(), "blob"), Total: 1,
			Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
		}
		if err := download.newPart(0, 1); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		err := download.run(ctx, requestURL, &registryOptions{Insecure: insecure})
		cancel()
		if insecure {
			if err != nil || hits.Load() != 1 {
				t.Fatalf("insecure download: error = %v, requests = %d, want nil, 1", err, hits.Load())
			}
		} else if err == nil || !strings.Contains(err.Error(), "not allowed") || hits.Load() != 0 {
			t.Fatalf("download: error = %v, requests = %d, want rejection before request", err, hits.Load())
		}
	}
}

func BenchmarkDownloadChunkCompletion(b *testing.B) {
	data := make([]byte, 1024*1024)
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprint(len(data)))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(data)
	}))
	b.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		b.Fatal(err)
	}
	downloadPath := filepath.Join(b.TempDir(), "blob")

	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		download := &blobDownload{Name: downloadPath, Digest: digest}
		part := &blobDownloadPart{Size: int64(len(data)), blobDownload: download}
		if err := download.downloadChunk(b.Context(), server.Client(), requestURL, io.Discard, part); err != nil {
			b.Fatal(err)
		}
	}
}

func TestDownloadChunkReturnsWhenTransferCompletes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte{0})
	}))
	t.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}

	download := &blobDownload{
		Name:   filepath.Join(t.TempDir(), "blob"),
		Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
	}
	part := &blobDownloadPart{Size: 1, blobDownload: download}
	ctx, cancel := context.WithTimeout(t.Context(), 250*time.Millisecond)
	defer cancel()

	if err := download.downloadChunk(ctx, server.Client(), requestURL, io.Discard, part); err != nil {
		t.Fatalf("downloadChunk() error = %v, want nil", err)
	}
}

func TestDownloadChunkDetectsStallBeforeFirstByte(t *testing.T) {
	requestStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(requestStarted)
		w.Header().Set("Content-Length", "1")
		w.WriteHeader(http.StatusPartialContent)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(server.Close)

	requestURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}

	download := &blobDownload{Digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"}
	part := &blobDownloadPart{Size: 1, blobDownload: download}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()

	originalStallTimeout := downloadStallTimeout
	downloadStallTimeout = 50 * time.Millisecond
	t.Cleanup(func() {
		downloadStallTimeout = originalStallTimeout
	})

	started := time.Now()
	err = download.downloadChunk(ctx, server.Client(), requestURL, io.Discard, part)
	elapsed := time.Since(started)

	select {
	case <-requestStarted:
	default:
		t.Fatal("download request did not start")
	}
	if !errors.Is(err, errPartStalled) {
		t.Fatalf("downloadChunk() error = %v after %v, want %v", err, elapsed, errPartStalled)
	}
	if elapsed >= 5*downloadStallTimeout {
		t.Fatalf("downloadChunk() detected the stall after %v, want less than %v", elapsed, 5*downloadStallTimeout)
	}
}
