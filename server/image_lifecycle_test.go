package server

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/imagegen"
	"github.com/stretchr/testify/require"
)

func imageLifecycleJob(service *imageService, id string) *imageJob {
	return &imageJob{ID: id, Kind: "generate", Config: cloneImageConfig(service.config), Profile: service.config.Profiles[0], Request: imageJobRequest{Prompt: "lifecycle fixture"}, ctx: service.ctx}
}

func TestImageQueuedRuntimePreflightDoesNotUnloadResidentText(t *testing.T) {
	_, service, _, events, runner := newImageTestService(t)
	runner.status = &imagegen.RunnerStatus{Ready: false, Error: "runtime removed while the image waited in queue"}
	require.ErrorContains(t, service.generate(imageLifecycleJob(service, "runtime-disappeared")), "runtime removed")
	require.Equal(t, []string{"close_image"}, events.snapshot())
	release, err := service.gate.reader(t.Context())
	require.NoError(t, err)
	release()
}

func TestImagePreparedRuntimeIsClosedWhenTransitionIsCancelled(t *testing.T) {
	_, service, _, events, _ := newImageTestService(t)
	ctx, cancel := context.WithCancel(service.ctx)
	cancel()
	job := imageLifecycleJob(service, "cancelled-before-unload")
	job.ctx = ctx
	require.ErrorIs(t, service.generate(job), context.Canceled)
	require.Equal(t, []string{"close_image"}, events.snapshot())
	service.gate.mu.Lock()
	waiting := len(service.gate.writers)
	service.gate.mu.Unlock()
	require.Zero(t, waiting)
}

type imageRecoveryText struct {
	*imageTestText
	free       []uint64
	queries    int
	restoreErr error
}

type imageBlockedRestore struct {
	*imageTestText
	entered chan struct{}
	release <-chan struct{}
}

func (b *imageBlockedRestore) Restore(ctx context.Context, states []imageTextState) []string {
	close(b.entered)
	select {
	case <-b.release:
		return b.imageTestText.Restore(ctx, states)
	case <-ctx.Done():
		return []string{ctx.Err().Error()}
	}
}

func TestImageTextAdmissionRemainsClosedUntilRestoreCompletes(t *testing.T) {
	_, service, _, events, _ := newImageTestService(t)
	releaseRestore := make(chan struct{})
	text := &imageBlockedRestore{imageTestText: &imageTestText{events: events}, entered: make(chan struct{}), release: releaseRestore}
	service.text = text
	finished := make(chan error, 1)
	go func() { finished <- service.generate(imageLifecycleJob(service, "restore-before-chat")) }()
	select {
	case <-text.entered:
	case <-time.After(time.Second):
		t.Fatal("restoration did not begin")
	}
	readers := make(chan func(), 1)
	go func() {
		release, err := service.gate.reader(service.ctx)
		if err == nil {
			readers <- release
		}
	}()
	require.Eventually(t, func() bool {
		service.gate.mu.Lock()
		defer service.gate.mu.Unlock()
		return service.gate.waitingReaders == 1
	}, time.Second, time.Millisecond)
	require.Empty(t, readers)
	close(releaseRestore)
	require.NoError(t, <-finished)
	select {
	case release := <-readers:
		release()
	case <-time.After(time.Second):
		t.Fatal("queued text did not resume after restoration")
	}
}

func (b *imageRecoveryText) FreeVRAM(context.Context) (uint64, error) {
	b.events.add("free_memory")
	index := b.queries
	b.queries++
	if index >= len(b.free) {
		index = len(b.free) - 1
	}
	return b.free[index], nil
}
func (b *imageRecoveryText) Restore(ctx context.Context, states []imageTextState) []string {
	b.restoreErr = ctx.Err()
	return b.imageTestText.Restore(ctx, states)
}

func TestImageRestoreWaitsForReleasedDriverMemory(t *testing.T) {
	_, service, _, events, _ := newImageTestService(t)
	text := &imageRecoveryText{imageTestText: &imageTestText{events: events}, free: []uint64{16 << 30, 8 << 30, 16 << 30}}
	service.text = text
	require.NoError(t, service.generate(imageLifecycleJob(service, "delayed-driver-free")))
	require.Equal(t, []string{"suspend_text", "free_memory", "run_image", "close_image", "free_memory", "free_memory", "restore_text"}, events.snapshot())
	require.NoError(t, text.restoreErr)
}

func TestImageRecoveryTimeoutStillAttemptsRestoreWithLiveContext(t *testing.T) {
	_, service, _, events, _ := newImageTestService(t)
	text := &imageRecoveryText{imageTestText: &imageTestText{events: events}, free: []uint64{16 << 30, 8 << 30}}
	service.text = text
	require.NoError(t, service.generate(imageLifecycleJob(service, "other-program-memory")))
	require.Contains(t, events.snapshot(), "restore_text")
	require.NoError(t, text.restoreErr, "the short recovery deadline must not cancel restoration")
}

func TestImageTransientKeepAliveDoesNotPreloadButNextTextRequestStillQueues(t *testing.T) {
	scheduler := InitScheduler(t.Context())
	backend := &schedulerImageText{server: &Server{sched: scheduler}}
	model := &Model{Name: "transient-model", Digest: "transient-digest"}
	options := api.DefaultOptions()
	options.NumCtx = 4
	require.Empty(t, backend.Restore(t.Context(), []imageTextState{{model: model, options: options, duration: 0}}))
	require.Empty(t, scheduler.pendingReqCh)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	_, failed := scheduler.getRunner(ctx, model, options, &api.Duration{Duration: 0}, false, false, nil)
	require.Empty(t, failed)
	require.Len(t, scheduler.pendingReqCh, 1, "the ordinary next turn retains its normal model-load path")
}

type imageTransientText struct{ *imageTestText }

func (b *imageTransientText) Suspend(ctx context.Context) ([]imageTextState, error) {
	states, err := b.imageTestText.Suspend(ctx)
	for i := range states {
		states[i].duration = 0
	}
	return states, err
}

func TestImageTransientModelsDoNotDelayCompletionForUnusedRestore(t *testing.T) {
	_, service, _, events, _ := newImageTestService(t)
	service.text = &imageTransientText{imageTestText: &imageTestText{events: events}}
	require.NoError(t, service.generate(imageLifecycleJob(service, "transient-model")))
	require.Equal(t, []string{"suspend_text", "free_memory", "run_image", "close_image"}, events.snapshot())
}

func TestImageCompletedAssetSurvivesRunnerErrorOrLateCancellation(t *testing.T) {
	for _, lateCancel := range []bool{false, true} {
		t.Run(map[bool]string{false: "runner-error", true: "late-cancel"}[lateCancel], func(t *testing.T) {
			_, service, router, _, runner := newImageTestService(t)
			job := imageLifecycleJob(service, "saved-result")
			runErr := errors.New("runner reported an error after saving")
			if lateCancel {
				ctx, cancel := context.WithCancel(service.ctx)
				defer cancel()
				job.ctx = ctx
				runner.afterSave = cancel
				runErr = context.Canceled
			}
			runner.resultErr = runErr
			require.ErrorIs(t, service.generate(job), runErr)
			require.NotNil(t, job.Result)
			_, err := os.Stat(job.Result.Path)
			require.NoError(t, err)
			response := imageTestHTTP(router, "GET", "/api/images/assets/"+job.Result.AssetID, nil, "")
			require.Equal(t, 200, response.Code)
			require.Empty(t, job.RestoreErrors)
		})
	}
}

func TestImageFailedWorkerPublishesItsCompletedAsset(t *testing.T) {
	_, service, router, _, runner := newImageTestService(t)
	runner.resultErr = errors.New("runner exited after writing the complete image")
	job := imageLifecycleJob(service, "")
	job.ctx = nil
	job, err := service.enqueue(job)
	require.NoError(t, err)
	require.Eventually(t, func() bool { service.mu.Lock(); defer service.mu.Unlock(); return imageTerminal(job.Status) }, 2*time.Second, time.Millisecond)
	response := imageTestHTTP(router, "GET", "/api/images/jobs/"+job.ID, nil, "")
	var result struct {
		Job imageJob `json:"job"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &result))
	require.Equal(t, "failed", result.Job.Status)
	require.NotNil(t, result.Job.Result)
	require.Contains(t, result.Job.Error, "complete image")
}
