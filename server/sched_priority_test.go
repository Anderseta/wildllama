package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/require"
)

func priorityFixture(ctx context.Context, name string, background bool, at time.Time) *LlmRequest {
	options := api.DefaultOptions()
	options.NumCtx = 32
	options.NumGPU = 0
	return &LlmRequest{ctx: ctx, model: &Model{ModelPath: name}, opts: options, background: background, queuedAt: at,
		successCh: make(chan *runnerRef, 1), errCh: make(chan error, 1), contextShift: true}
}

func TestSchedulerPriorityBoundedBurstAndAging(t *testing.T) {
	now := time.Now()
	for _, aged := range []bool{false, true} {
		t.Run(map[bool]string{false: "burst", true: "aged"}[aged], func(t *testing.T) {
			queue := &pendingModelQueue{}
			at := now
			if aged {
				at = now.Add(-pendingAgingTime)
			}
			background := priorityFixture(t.Context(), "background", true, at)
			queue.add(background, now)
			for range 6 {
				queue.add(priorityFixture(t.Context(), "interactive", false, now), now)
			}
			beforeBackground := 0
			for len(queue.requests) > 0 {
				if queue.take(now, nil) == background {
					break
				}
				beforeBackground++
			}
			if aged {
				require.Equal(t, 1, beforeBackground)
			} else {
				require.Equal(t, pendingInteractiveBurst, beforeBackground)
			}
		})
	}
}

func TestSchedulerPriorityBoundedCompatibleGrouping(t *testing.T) {
	now := time.Now()
	first := priorityFixture(t.Context(), "A", false, now)
	queue := &pendingModelQueue{previous: first}
	oldest := priorityFixture(t.Context(), "B", false, now)
	second := priorityFixture(t.Context(), "A", false, now)
	third := priorityFixture(t.Context(), "A", false, now)
	queue.requests = []*LlmRequest{oldest, second, third}
	resident := &loadedRunnerSnapshot{byKey: map[string]*runnerRef{"A": {}}}
	require.Same(t, second, queue.take(now, resident))
	require.Same(t, oldest, queue.take(now, resident), "a resident model may not repeatedly overtake an older request")
	require.Same(t, third, queue.take(now, resident))

	changed := priorityFixture(t.Context(), "A", false, now)
	changed.opts.NumCtx = 64
	queue = &pendingModelQueue{previous: first, requests: []*LlmRequest{oldest, changed, second}}
	require.Same(t, oldest, queue.take(now, resident), "a queued model configuration change fences later reuse")
	queue = &pendingModelQueue{previous: first, requests: []*LlmRequest{oldest, second}}
	oldest.queuedAt = now.Add(-pendingAgingTime)
	require.Same(t, oldest, queue.take(now, resident), "an aged request may not be bypassed for cache affinity")
}

func TestSchedulerGroupingNeverChangesMemoryDraftOrAdapterRequirements(t *testing.T) {
	now := time.Now()
	first := priorityFixture(t.Context(), "A", false, now)
	for _, change := range []struct {
		name  string
		apply func(*LlmRequest)
	}{
		{"context", func(r *LlmRequest) { r.opts.NumCtx = 64 }},
		{"batch", func(r *LlmRequest) { r.opts.NumBatch++ }},
		{"vram", func(r *LlmRequest) { v := 256; r.opts.GPUVRAMLimitMiB = &v }},
		{"ram", func(r *LlmRequest) { r.opts.RunnerEnv = map[string]string{"LLAMA_ARG_CACHE_RAM": "64"} }},
		{"draft", func(r *LlmRequest) { r.model.DraftPath = "different" }},
		{"adapter", func(r *LlmRequest) { r.model.AdapterPaths = []string{"different"} }},
		{"shift", func(r *LlmRequest) { r.contextShift = false }},
	} {
		t.Run(change.name, func(t *testing.T) {
			next := priorityFixture(t.Context(), "A", false, now)
			change.apply(next)
			require.False(t, compatiblePendingModels(first, next))
		})
	}
}

func TestSchedulerReorderWindowRetainsOriginalAdmissionBoundAndCancellation(t *testing.T) {
	t.Setenv("OLLAMA_MAX_QUEUE", "2")
	scheduler := InitScheduler(t.Context())
	options := api.DefaultOptions()
	options.NumGPU = 0
	options.NumCtx = 32
	firstCtx, cancelFirst := context.WithCancel(t.Context())
	defer cancelFirst()
	_, failed := scheduler.getRunner(firstCtx, &Model{ModelPath: "A"}, options, nil, false, false, nil)
	require.Empty(t, failed)
	_, failed = scheduler.getRunner(t.Context(), &Model{ModelPath: "B"}, options, nil, false, false, nil)
	require.Empty(t, failed)
	queue := &pendingModelQueue{}
	defer scheduler.discardPendingQueue(queue)
	cancelFirst()
	selected := scheduler.nextPendingRequest(t.Context(), queue)
	require.Equal(t, "B", selected.model.ModelPath)
	require.Zero(t, scheduler.pendingOwned.Load(), "cancelled queued work releases its admission slot")
	for _, name := range []string{"C", "D"} {
		_, failed = scheduler.getRunner(t.Context(), &Model{ModelPath: name}, options, nil, false, false, nil)
		require.Empty(t, failed)
	}
	selected = scheduler.nextPendingRequest(t.Context(), queue)
	require.Equal(t, "C", selected.model.ModelPath)
	require.Len(t, queue.requests, 1)
	require.Empty(t, scheduler.pendingReqCh)
	_, failed = scheduler.getRunner(t.Context(), &Model{ModelPath: "E"}, options, nil, false, false, nil)
	require.Empty(t, failed)
	_, failed = scheduler.getRunner(t.Context(), &Model{ModelPath: "F"}, options, nil, false, false, nil)
	require.ErrorIs(t, <-failed, ErrMaxQueue)
	require.Equal(t, int64(2), scheduler.pendingOwned.Load(), "channel plus reorder window remain bounded")
}

func TestSchedulerPrioritySelectsChatBeforeQueuedBackgroundLoad(t *testing.T) {
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "1")
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	scheduler := InitScheduler(ctx)
	scheduler.getSystemInfoFn = func() ml.SystemInfo { return ml.SystemInfo{} }
	order := make(chan string, 4)
	scheduler.loadFn = func(request *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
		order <- request.model.ModelPath
		return false
	}
	background := priorityFixture(ctx, "index-model", true, time.Now())
	chat := priorityFixture(ctx, "chat-model", false, time.Now())
	scheduler.pendingReqCh <- background
	scheduler.pendingReqCh <- chat
	done := make(chan struct{})
	go func() { defer close(done); scheduler.processPending(ctx) }()
	select {
	case first := <-order:
		require.Equal(t, "chat-model", first)
	case <-time.After(time.Second):
		t.Fatal("scheduler did not select chat")
	}
	select {
	case second := <-order:
		require.Equal(t, "index-model", second)
	case <-time.After(time.Second):
		t.Fatal("background request was starved")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("scheduler did not stop")
	}
}

func TestSchedulerShutdownRejectsBacklogAndIncomingWithoutLeakingQueueSlots(t *testing.T) {
	t.Setenv("OLLAMA_MAX_QUEUE", "2")
	scheduler := InitScheduler(t.Context())
	options := api.DefaultOptions()
	options.NumCtx = 32
	options.NumGPU = 0
	_, active := scheduler.getRunner(t.Context(), &Model{ModelPath: "A"}, options, nil, false, false, nil)
	_, backlog := scheduler.getRunner(t.Context(), &Model{ModelPath: "B"}, options, nil, false, false, nil)
	queue := &pendingModelQueue{}
	require.Equal(t, "A", scheduler.nextPendingRequest(t.Context(), queue).model.ModelPath)
	_, incoming := scheduler.getRunner(t.Context(), &Model{ModelPath: "C"}, options, nil, false, false, nil)
	scheduler.discardPendingQueue(queue)
	require.Empty(t, active, "active native work is handled separately from queued work")
	require.ErrorIs(t, <-backlog, context.Canceled)
	require.ErrorIs(t, <-incoming, context.Canceled)
	require.Zero(t, scheduler.pendingOwned.Load())
	require.Zero(t, scheduler.pendingBacklog.Load())
	require.Empty(t, scheduler.pendingReqCh)
}

func TestInferencePriorityReachesSchedulerThroughRequestContext(t *testing.T) {
	for _, test := range []struct {
		path, header string
		interactive  bool
	}{{"/api/chat", "", true}, {"/api/embed", "", false}, {"/api/chat", "background", false}, {"/api/embed", "interactive", true}} {
		t.Run(test.path+test.header, func(t *testing.T) {
			server := &Server{}
			router := gin.New()
			router.Use(server.inferenceGuard())
			router.POST(test.path, func(c *gin.Context) {
				interactive, specified := api.RequestPriority(c.Request.Context())
				c.JSON(200, gin.H{"interactive": interactive, "specified": specified})
			})
			request := httptest.NewRequest(http.MethodPost, test.path, nil)
			request.Header.Set("X-Wildgoose-Priority", test.header)
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			var value struct {
				Interactive bool `json:"interactive"`
				Specified   bool `json:"specified"`
			}
			require.NoError(t, json.Unmarshal(response.Body.Bytes(), &value))
			require.True(t, value.Specified)
			require.Equal(t, test.interactive, value.Interactive)
		})
	}
}
