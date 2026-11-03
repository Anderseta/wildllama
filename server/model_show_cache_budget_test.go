package server

import (
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/ollama/ollama/api"
	modelpkg "github.com/ollama/ollama/types/model"
	"github.com/stretchr/testify/require"
)

func assertShowCacheBudget(t *testing.T, cache *modelShowCache) {
	t.Helper()
	cache.mu.Lock()
	defer cache.mu.Unlock()
	require.LessOrEqual(t, len(cache.local)+len(cache.cloud), cache.maxEntries)
	bytes := 0
	for _, entry := range cache.local {
		bytes += entry.Bytes
	}
	for _, usage := range cache.cloudUsage {
		bytes += usage.Bytes
	}
	require.Equal(t, bytes, cache.bytes)
	require.LessOrEqual(t, cache.bytes, cache.maxBytes)
	require.Len(t, cache.cloudUsage, len(cache.cloud))
}

func TestModelShowCacheSharedByteBudgetEvictsLeastRecentlyUsed(t *testing.T) {
	cache := newModelShowCache()
	response := showCacheTestResponse(1, false)
	cost, fits := modelShowResponseBytes(response, 1<<20)
	require.True(t, fits)
	cache.maxBytes = 2 * (cost + 129)
	a, b, c := modelShowLocalKey{Model: "a"}, modelShowCloudKey{Model: "b"}, modelShowLocalKey{Model: "c"}
	cache.setLocal(a, "", response)
	cache.setCloud(b, response)
	_, ok := cache.getLocal(a, "")
	require.True(t, ok)
	cache.setLocal(c, "", response)
	_, ok = cache.getCloud(b)
	require.False(t, ok, "older cloud entry should share the local byte budget")
	_, ok = cache.getLocal(a, "")
	require.True(t, ok)
	_, ok = cache.getLocal(c, "")
	require.True(t, ok)
	assertShowCacheBudget(t, cache)
}

func TestModelShowCacheSharedEntryBudgetAndInvalidationReleaseBytes(t *testing.T) {
	cache := newModelShowCache()
	cache.maxEntries = 2
	response := showCacheTestResponse(1, false)
	name := modelpkg.ParseName("budget-local")
	a, b, c := modelShowLocalKey{Model: name.String()}, modelShowCloudKey{Model: "b"}, modelShowCloudKey{Model: "c"}
	cache.setLocal(a, "one", response)
	cache.setCloud(b, response)
	_, ok := cache.getLocal(a, "one")
	require.True(t, ok)
	cache.setCloud(c, response)
	_, ok = cache.getCloud(b)
	require.False(t, ok)
	cache.invalidateLocal(name)
	_, ok = cache.getLocal(a, "one")
	require.False(t, ok)
	assertShowCacheBudget(t, cache)
	cache.setCloud(c, response)
	assertShowCacheBudget(t, cache)
	cache.removeCloudForTest(c)
	require.Zero(t, cache.bytes)
	cache.endCloudReadRefresh(c)
	require.Empty(t, cache.cloudNextReadRefreshAfter)
}

func (c *modelShowCache) removeCloudForTest(key modelShowCloudKey) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.removeCloudLocked(key)
}

func TestModelShowCacheOversizedResponseBypassesWithoutTruncation(t *testing.T) {
	setTestHome(t, t.TempDir())
	createShowCacheModel(t, "large-show-response", map[string]any{"test.context_length": uint32(1024)})
	cache := newModelShowCache()
	cache.maxBytes = 16 << 10
	small := modelShowCloudKey{Model: "small"}
	cache.setCloud(small, showCacheTestResponse(1, false))
	before := cache.bytes
	large := &api.ShowResponse{License: strings.Repeat("full-license-", 8192), ModelInfo: map[string]any{"vocabulary": []string{strings.Repeat("token", 10000)}}}
	calls := 0
	cache.getModelInfo = func(api.ShowRequest) (*api.ShowResponse, error) { calls++; return large, nil }
	for i := 0; i < 2; i++ {
		response, err := cache.GetLocal(api.ShowRequest{Model: "large-show-response", Verbose: true})
		require.NoError(t, err)
		require.Equal(t, large, response)
	}
	require.Equal(t, 2, calls, "large metadata must be returned fully instead of retained")
	require.Empty(t, cache.local)
	require.Equal(t, before, cache.bytes, "bypass must not evict useful small entries")
	_, ok := cache.getCloud(small)
	require.True(t, ok)
	cache.setCloud(modelShowCloudKey{Model: "large"}, large)
	require.Len(t, cache.cloud, 1)
	assertShowCacheBudget(t, cache)
}

func TestModelShowCacheBudgetConcurrentAccess(t *testing.T) {
	cache := newModelShowCache()
	cache.maxEntries, cache.maxBytes = 3, 32<<10
	var workers sync.WaitGroup
	for i := 0; i < 16; i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for j := 0; j < 20; j++ {
				local := modelShowLocalKey{Model: fmt.Sprint((i + j) % 8)}
				cloud := modelShowCloudKey{Model: local.Model}
				cache.setLocal(local, "digest", showCacheTestResponse(j, false))
				cache.setCloud(cloud, showCacheTestResponse(j, false))
				cache.getLocal(local, "digest")
				cache.getCloud(cloud)
			}
		}()
	}
	workers.Wait()
	assertShowCacheBudget(t, cache)
}
