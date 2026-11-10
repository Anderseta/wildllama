package server

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	modelpkg "github.com/ollama/ollama/types/model"
)

func TestModelShowCacheCancellationReleasesOnlyCanceledWaiter(t *testing.T) {
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "cancel-show", map[string]any{"test.context_length": uint32(1024)})
	cache := newModelShowCache()
	started, release := make(chan struct{}), make(chan struct{})
	t.Cleanup(func() {
		select {
		case <-release:
		default:
			close(release)
		}
	})
	var calls atomic.Int32
	cache.getModelInfo = func(api.ShowRequest) (*api.ShowResponse, error) {
		if calls.Add(1) == 1 {
			close(started)
		}
		<-release
		return showCacheTestResponse(1, false), nil
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	first := make(chan error, 1)
	req := api.ShowRequest{Model: "cancel-show"}
	go func() { _, err := cache.GetLocalContext(ctx, req); first <- err }()
	<-started
	cancel()
	select {
	case err := <-first:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled lookup returned %v", err)
		}
	case <-time.After(300 * time.Millisecond):
		t.Fatal("canceled request remained blocked on the shared metadata parse")
	}
	second := make(chan *api.ShowResponse, 1)
	secondError := make(chan error, 1)
	go func() {
		response, err := cache.GetLocalContext(t.Context(), req)
		second <- response
		secondError <- err
	}()
	close(release)
	if err := <-secondError; err != nil {
		t.Fatal(err)
	}
	response := <-second
	response.ModelInfo["call"] = 99
	warm, err := cache.GetLocal(req)
	if err != nil || warm.ModelInfo["call"] != 1 || calls.Load() != 1 {
		t.Fatalf("shared work was lost or its cache mutated: calls=%d error=%v", calls.Load(), err)
	}
}

func TestModelShowCachePreCanceledLookupDoesNoModelWork(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	cache := newModelShowCache()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	cache.getModelInfo = func(api.ShowRequest) (*api.ShowResponse, error) {
		t.Fatal("canceled lookup started metadata work")
		return nil, nil
	}
	if _, err := cache.GetLocalContext(ctx, api.ShowRequest{Model: "not-installed"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled lookup returned %v", err)
	}
}

func TestModelShowCacheSingleMissKeepsCacheIndependent(t *testing.T) {
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "owned-show", map[string]any{"test.context_length": uint32(1024)})
	cache := newModelShowCache()
	cache.getModelInfo = func(api.ShowRequest) (*api.ShowResponse, error) {
		return &api.ShowResponse{
			ModelInfo: map[string]any{"vocabulary": []string{"original"}},
			Tensors:   []api.Tensor{{Name: "weight", Shape: []uint64{1, 2}}},
			Details:   api.ModelDetails{Families: []string{"family"}},
		}, nil
	}
	req := api.ShowRequest{Model: "owned-show"}
	response, err := cache.GetLocal(req)
	if err != nil {
		t.Fatal(err)
	}
	response.ModelInfo["vocabulary"].([]string)[0] = "caller changed"
	response.Tensors[0].Shape[0] = 99
	response.Details.Families[0] = "caller family"
	warm, err := cache.GetLocal(req)
	if err != nil || warm.ModelInfo["vocabulary"].([]string)[0] != "original" || warm.Tensors[0].Shape[0] != 1 || warm.Details.Families[0] != "family" {
		t.Fatalf("single-result ownership escaped into the cache: %+v error=%v", warm, err)
	}
}

func TestModelShowCacheAsyncPanicRemainsRequestFailure(t *testing.T) {
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "panic-show", map[string]any{"test.context_length": uint32(1024)})
	cache := newModelShowCache()
	cache.getModelInfo = func(api.ShowRequest) (*api.ShowResponse, error) { panic("fixture private cause") }
	req := api.ShowRequest{Model: "panic-show"}
	if _, err := cache.GetLocalContext(t.Context(), req); err == nil || err.Error() != "model metadata lookup failed" {
		t.Fatalf("async metadata panic was not scoped to its request: %v", err)
	}
	cache.mu.RLock()
	entries := len(cache.local)
	cache.mu.RUnlock()
	if entries != 0 {
		t.Fatal("panicked metadata lookup published a cached response")
	}
	cache.getModelInfo = func(api.ShowRequest) (*api.ShowResponse, error) { return showCacheTestResponse(2, false), nil }
	response, err := cache.GetLocalContext(t.Context(), req)
	if err != nil || response.ModelInfo["call"] != 2 {
		t.Fatalf("failed lookup prevented a subsequent retry: response=%+v error=%v", response, err)
	}
}

func BenchmarkModelShowSingleMissCopies(b *testing.B) {
	b.Setenv("OLLAMA_MODELS", b.TempDir())
	name := modelpkg.ParseName("bench-owned-show")
	if err := manifest.WriteManifestWithMetadata(name, manifest.Layer{Digest: "sha256:" + strings.Repeat("a", 64), Size: 1}, nil, manifest.RunnerLlamaCPP, manifest.FormatGGUF); err != nil {
		b.Fatal(err)
	}
	fixture := &api.ShowResponse{ModelInfo: map[string]any{"vocabulary": make([]string, 32768)}, Tensors: make([]api.Tensor, 1024)}
	for i := range fixture.ModelInfo["vocabulary"].([]string) {
		fixture.ModelInfo["vocabulary"].([]string)[i] = "token value"
	}
	for i := range fixture.Tensors {
		fixture.Tensors[i] = api.Tensor{Name: "weight", Shape: []uint64{1024, 1024}}
	}
	cache := newModelShowCache()
	cache.getModelInfo = func(api.ShowRequest) (*api.ShowResponse, error) { return cloneShowResponse(fixture), nil }
	req := api.ShowRequest{Model: name.String()}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		cache.invalidateLocal(name)
		if _, err := cache.GetLocal(req); err != nil {
			b.Fatal(err)
		}
	}
}
