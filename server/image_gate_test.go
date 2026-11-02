package server

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/require"
)

func TestImageGateDrainsExistingStreamsBeforeWriterAndQueuedReaders(t *testing.T) {
	gate := newImageGPUGate()
	reader, err := gate.reader(t.Context())
	require.NoError(t, err)
	writers := make(chan func(), 1)
	go func() {
		release, err := gate.exclusive(t.Context(), "image")
		if err == nil {
			writers <- release
		}
	}()
	require.Eventually(t, func() bool { gate.mu.Lock(); defer gate.mu.Unlock(); return len(gate.writers) == 1 }, time.Second, time.Millisecond)
	newReaders := make(chan func(), 1)
	go func() {
		release, err := gate.reader(t.Context())
		if err == nil {
			newReaders <- release
		}
	}()
	require.Never(t, func() bool { return len(writers) > 0 || len(newReaders) > 0 }, 20*time.Millisecond, time.Millisecond)
	reader()
	reader() // a completed/cancelled stream releases only once
	var writer func()
	select {
	case writer = <-writers:
	case <-time.After(time.Second):
		t.Fatal("image did not acquire drained GPU")
	}
	require.Empty(t, newReaders)
	writer()
	writer()
	select {
	case release := <-newReaders:
		release()
	case <-time.After(time.Second):
		t.Fatal("queued text did not resume")
	}
	gate.mu.Lock()
	require.Zero(t, gate.readers)
	gate.mu.Unlock()
}

func TestImageGateCancelledWriterCannotBlockReaders(t *testing.T) {
	gate := newImageGPUGate()
	reader, err := gate.reader(t.Context())
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { _, err := gate.exclusive(ctx, "cancelled"); done <- err }()
	require.Eventually(t, func() bool { gate.mu.Lock(); defer gate.mu.Unlock(); return len(gate.writers) == 1 }, time.Second, time.Millisecond)
	cancel()
	require.ErrorIs(t, <-done, context.Canceled)
	reader()
	next, err := gate.reader(t.Context())
	require.NoError(t, err)
	next()
	gate.mu.Lock()
	require.Empty(t, gate.writers)
	gate.mu.Unlock()
}

func TestImageGateSharedLeaseOwnerAndShutdown(t *testing.T) {
	gate := newImageGPUGate()
	release, err := gate.exclusive(t.Context(), "owner")
	require.NoError(t, err)
	require.Error(t, gate.coexist("other"))
	require.NoError(t, gate.coexist("owner"))
	reader, err := gate.reader(t.Context())
	require.NoError(t, err)
	reader()
	require.True(t, gate.shared.Load())
	release()
	require.False(t, gate.shared.Load())
	gate.shutdown()
	_, err = gate.reader(t.Context())
	require.ErrorIs(t, err, context.Canceled)
	_, err = gate.exclusive(t.Context(), "next")
	require.ErrorIs(t, err, context.Canceled)
}

func TestImageGateIdleRequestsReuseNotificationChannel(t *testing.T) {
	gate := newImageGPUGate()
	initial := gate.changed
	for i := 0; i < 32; i++ {
		release, err := gate.reader(t.Context())
		require.NoError(t, err)
		release()
	}
	require.Equal(t, initial, gate.changed, "ordinary inference must not allocate notification channels")
}

func TestImageGateCancelsRequestBeforeItsReaderLeaseIsReleased(t *testing.T) {
	server, service, _, _, _ := newImageTestService(t)
	var requestContext context.Context
	router := gin.New()
	router.Use(server.imageInferenceGate())
	router.POST("/api/generate", func(c *gin.Context) { requestContext = c.Request.Context(); c.String(200, "done") })
	response := imageTestHTTP(router, http.MethodPost, "/api/generate", nil, "")
	require.Equal(t, 200, response.Code)
	require.ErrorIs(t, requestContext.Err(), context.Canceled)
	service.gate.mu.Lock()
	require.Zero(t, service.gate.readers)
	service.gate.mu.Unlock()
}

func TestImageSimultaneousSchedulerAllowsOnlyUnchangedResidentModel(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	scheduler := InitScheduler(ctx)
	gate := newImageGPUGate()
	gate.shared.Store(true)
	scheduler.imageGate.Store(gate)
	options := api.DefaultOptions()
	options.NumCtx = 4
	model := &Model{Name: "resident", Digest: "resident-digest"}
	native := &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}}
	runner := &runnerRef{modelKey: schedulerModelKey(model), llama: native, Options: &options, numParallel: 1}
	runner.model.Store(model)
	scheduler.loaded[runner.modelKey] = runner
	scheduler.publishLoadedSnapshotLocked()
	requestCtx, done := context.WithCancel(ctx)
	success, failed := scheduler.getRunner(requestCtx, model, options, nil, false, false, nil)
	select {
	case actual := <-success:
		require.Same(t, runner, actual)
	default:
		t.Fatal("healthy resident model was not reused")
	}
	require.Empty(t, failed)
	done()
	changed := options
	changed.NumCtx = 8
	for _, candidate := range []struct {
		model *Model
		opts  api.Options
	}{{model, changed}, {&Model{Name: "new", Digest: "new-digest"}, options}} {
		_, failed = scheduler.getRunner(ctx, candidate.model, candidate.opts, nil, false, false, nil)
		select {
		case err := <-failed:
			require.ErrorIs(t, err, errImageResidentOnly)
		default:
			t.Fatal("memory-changing/new load was not rejected")
		}
	}
	require.Empty(t, scheduler.pendingReqCh)
	require.False(t, native.closeCalled.Load())
	require.Equal(t, 4, runner.Options.NumCtx)
}

type imageClosingRunner struct {
	mockLlm
	entered, finish chan struct{}
}

func (r *imageClosingRunner) Close() error { close(r.entered); <-r.finish; return nil }

func TestImageWaitsForCancelledLoaderNativeCleanup(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	scheduler := InitScheduler(ctx)
	native := &imageClosingRunner{mockLlm: mockLlm{modelPath: "cancelled-loader"}, entered: make(chan struct{}), finish: make(chan struct{})}
	scheduler.activeLoading = native
	scheduler.activeStatus = &loadedModel{loading: true}
	scheduler.publishLoadedSnapshotLocked()
	drained := make(chan error, 1)
	go func() { drained <- waitForImageTextLoaders(ctx, scheduler) }()
	require.Never(t, func() bool { return len(drained) > 0 }, 20*time.Millisecond, time.Millisecond)
	cleanup := make(chan struct{})
	go func() {
		scheduler.discardPendingLoading(&LlmRequest{model: &Model{ModelPath: "cancelled-loader"}})
		close(cleanup)
	}()
	<-native.entered
	require.Nil(t, scheduler.loadedSnapshot.Load().loading, "registry clears before native Close completes")
	require.Never(t, func() bool { return len(drained) > 0 }, 30*time.Millisecond, time.Millisecond)
	close(native.finish)
	<-cleanup
	require.NoError(t, <-drained)
}

func TestImageLoaderDrainHonorsContextAndLoadedRunnerInitialization(t *testing.T) {
	scheduler := InitScheduler(t.Context())
	runner := &runnerRef{}
	runner.status.Store(&loadedModel{loading: true})
	scheduler.loadedSnapshot.Store(&loadedRunnerSnapshot{runners: []*runnerRef{runner}})
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Millisecond)
	defer cancel()
	require.ErrorIs(t, waitForImageTextLoaders(ctx, scheduler), context.DeadlineExceeded)
	runner.status.Store(&loadedModel{loading: false})
	require.NoError(t, waitForImageTextLoaders(t.Context(), scheduler))
}
