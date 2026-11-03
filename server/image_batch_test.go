package server

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
	"github.com/stretchr/testify/require"
)

type imageBatchTestRunner struct {
	base              *imageTestRunner
	mu                sync.Mutex
	calls             []imagegen.Request
	active, maxActive int
	blockAt           int
	blocked           chan struct{}
	release           <-chan struct{}
	errorAfter        int
	afterSave         func(int)
	closeEntered      chan struct{}
	closeRelease      chan struct{}
	closeOnce         sync.Once
}

func (b *imageBatchTestRunner) Status() imagegen.RunnerStatus { return b.base.Status() }
func (b *imageBatchTestRunner) Close() error {
	if b.closeEntered != nil {
		b.closeOnce.Do(func() { close(b.closeEntered) })
		<-b.closeRelease
	}
	return b.base.Close()
}
func (b *imageBatchTestRunner) Run(ctx context.Context, profile imagegen.Profile, request imagegen.Request, progress func(imagegen.Progress)) (imagegen.Result, error) {
	b.mu.Lock()
	b.calls = append(b.calls, request)
	index := len(b.calls)
	b.active++
	if b.active > b.maxActive {
		b.maxActive = b.active
	}
	b.mu.Unlock()
	defer func() { b.mu.Lock(); b.active--; b.mu.Unlock() }()
	if index == b.blockAt {
		close(b.blocked)
		select {
		case <-b.release:
		case <-ctx.Done():
			return imagegen.Result{}, ctx.Err()
		}
	}
	result, err := b.base.Run(ctx, profile, request, progress)
	if b.afterSave != nil {
		b.afterSave(index)
	}
	if err == nil && index == b.errorAfter {
		return result, errors.New("runner error after completed batch image")
	}
	return result, err
}

func newImageBatchTestService(t *testing.T) (*Server, *imageService, *gin.Engine, *imageTestEvents, *imageBatchTestRunner) {
	t.Helper()
	server, service, router, events, base := newImageTestService(t)
	runner := &imageBatchTestRunner{base: base}
	service.newEngine = func(imagegen.Config) (imagegen.Runner, error) { return runner, nil }
	return server, service, router, events, runner
}

func waitImageBatchTerminal(t *testing.T, service *imageService, job *imageJob) {
	t.Helper()
	require.Eventually(t, func() bool { service.mu.Lock(); defer service.mu.Unlock(); return imageTerminal(job.Status) }, 3*time.Second, time.Millisecond)
}

func countImageEvent(events *imageTestEvents, value string) int {
	n := 0
	for _, event := range events.snapshot() {
		if event == value {
			n++
		}
	}
	return n
}

func TestImageBatchSequentialResultsUseOneTextTransition(t *testing.T) {
	_, service, router, events, runner := newImageBatchTestService(t)
	job := imageLifecycleJob(service, "")
	seed := int64(41)
	job.Request.Count = 3
	job.Request.Seed = &seed
	job, err := service.enqueue(job)
	require.NoError(t, err)
	waitImageBatchTerminal(t, service, job)
	service.mu.Lock()
	status, errorMessage := job.Status, job.Error
	results := append([]imageAsset(nil), job.Results...)
	first := *job.Result
	progress := job.Progress
	service.mu.Unlock()
	require.Equal(t, "completed", status, errorMessage)
	require.Len(t, results, 3)
	require.Equal(t, results[0], first)
	for i, asset := range results {
		require.Equal(t, int64(41+i), asset.Seed)
		require.Equal(t, i+1, asset.BatchIndex)
		require.Equal(t, 3, asset.BatchCount)
		require.Equal(t, imageBatchFilename(i+1), asset.Filename)
		response := imageTestHTTP(router, "GET", "/api/images/assets/"+asset.AssetID, nil, "")
		require.Equal(t, 200, response.Code)
	}
	require.Equal(t, 3, progress.BatchIndex)
	require.Equal(t, 3, progress.BatchCount)
	require.Equal(t, 1, countImageEvent(events, "suspend_text"))
	require.Equal(t, 1, countImageEvent(events, "restore_text"))
	require.Equal(t, 1, countImageEvent(events, "close_image"))
	runner.mu.Lock()
	active, calls := runner.maxActive, len(runner.calls)
	runner.mu.Unlock()
	require.Equal(t, 1, active)
	require.Equal(t, 3, calls)
}

func TestImageBatchResultsPersistWhileLaterImageIsRunningAndSurviveRestart(t *testing.T) {
	_, service, router, events, runner := newImageBatchTestService(t)
	runner.blockAt = 2
	runner.blocked = make(chan struct{})
	release := make(chan struct{})
	runner.release = release
	job := imageLifecycleJob(service, "")
	job.Request.Count = 3
	job.Request.DeferTextRestore = true
	job.SessionID = "session-a"
	job.Request.SessionID = job.SessionID
	job, err := service.enqueue(job)
	require.NoError(t, err)
	select {
	case <-runner.blocked:
	case <-time.After(time.Second):
		t.Fatal("second image did not start")
	}
	data, err := os.ReadFile(filepath.Join(service.root, "jobs.json"))
	require.NoError(t, err)
	var persisted []imageJob
	require.NoError(t, json.Unmarshal(data, &persisted))
	require.Len(t, persisted, 1)
	require.Len(t, persisted[0].Results, 1)
	require.Equal(t, "running", persisted[0].Status)
	require.Equal(t, 3, persisted[0].Generation.Count)
	t.Setenv("WILDLLAMA_IMAGES_DIR", service.root)
	restarted := &Server{}
	t.Cleanup(restarted.stopImageJobs)
	recovered := restarted.imageService()
	require.Len(t, recovered.jobs, 1)
	require.Equal(t, "interrupted", recovered.jobs[0].Status)
	require.Len(t, recovered.jobs[0].Results, 1)
	response := imageTestHTTP(mustImageRouter(t, restarted), "GET", "/api/images/jobs?session_id=session-a", nil, "")
	require.Equal(t, 200, response.Code)
	require.Contains(t, response.Body.String(), "image.png")
	response = imageTestHTTP(router, "POST", "/api/images/jobs/"+job.ID+"/cancel", nil, "")
	require.Equal(t, 200, response.Code)
	waitImageBatchTerminal(t, service, job)
	service.mu.Lock()
	status, deferred, count := job.Status, job.TextRestoreDeferred, len(job.Results)
	service.mu.Unlock()
	require.Equal(t, "cancelled", status)
	require.False(t, deferred)
	require.Equal(t, 1, count)
	require.Equal(t, 1, countImageEvent(events, "restore_text"))
}

func TestImageBatchDeferRestoreCommitsOnlyCompletedFullBatch(t *testing.T) {
	_, service, router, events, _ := newImageBatchTestService(t)
	job := imageLifecycleJob(service, "")
	job.Request.Count = 2
	job.Request.DeferTextRestore = true
	job, err := service.enqueue(job)
	require.NoError(t, err)
	waitImageBatchTerminal(t, service, job)
	service.mu.Lock()
	status, deferred, count := job.Status, job.TextRestoreDeferred, len(job.Results)
	service.mu.Unlock()
	require.Equal(t, "completed", status)
	require.True(t, deferred)
	require.Equal(t, 2, count)
	require.Zero(t, countImageEvent(events, "restore_text"))
	require.Equal(t, 1, countImageEvent(events, "suspend_text"))
	response := imageTestHTTP(router, "POST", "/api/images/jobs/"+job.ID+"/cancel", nil, "")
	require.Equal(t, 200, response.Code)
	service.mu.Lock()
	status, deferred = job.Status, job.TextRestoreDeferred
	service.mu.Unlock()
	require.Equal(t, "completed", status)
	require.True(t, deferred)
	release, err := service.gate.reader(t.Context())
	require.NoError(t, err)
	release()
}

func TestImageBatchErrorAfterFinalSaveStillRestoresAndRetainsAllResults(t *testing.T) {
	_, service, _, events, runner := newImageBatchTestService(t)
	runner.errorAfter = 2
	job := imageLifecycleJob(service, "")
	job.Request.Count = 2
	job.Request.DeferTextRestore = true
	job, err := service.enqueue(job)
	require.NoError(t, err)
	waitImageBatchTerminal(t, service, job)
	service.mu.Lock()
	status, deferred, count := job.Status, job.TextRestoreDeferred, len(job.Results)
	service.mu.Unlock()
	require.Equal(t, "failed", status)
	require.False(t, deferred)
	require.Equal(t, 2, count)
	require.Equal(t, 1, countImageEvent(events, "restore_text"))
}

func TestImageBatchCancelDuringCloseWinsBeforeDeferredCommit(t *testing.T) {
	_, service, router, events, runner := newImageBatchTestService(t)
	runner.closeEntered = make(chan struct{})
	runner.closeRelease = make(chan struct{})
	var releaseOnce sync.Once
	releaseClose := func() { releaseOnce.Do(func() { close(runner.closeRelease) }) }
	t.Cleanup(releaseClose)
	job := imageLifecycleJob(service, "")
	job.Request.DeferTextRestore = true
	job, err := service.enqueue(job)
	require.NoError(t, err)
	select {
	case <-runner.closeEntered:
	case <-time.After(time.Second):
		t.Fatal("engine cleanup did not begin")
	}
	response := imageTestHTTP(router, "POST", "/api/images/jobs/"+job.ID+"/cancel", nil, "")
	require.Equal(t, 200, response.Code)
	releaseClose()
	waitImageBatchTerminal(t, service, job)
	service.mu.Lock()
	status, deferred, count := job.Status, job.TextRestoreDeferred, len(job.Results)
	service.mu.Unlock()
	require.Equal(t, "cancelled", status)
	require.False(t, deferred)
	require.Equal(t, 1, count)
	require.Equal(t, 1, countImageEvent(events, "restore_text"))
}

func TestImageBatchExplicitGlobalNoRestoreRemainsAuthoritative(t *testing.T) {
	_, service, _, events, _ := newImageBatchTestService(t)
	job := imageLifecycleJob(service, "")
	job.Request.Count = 2
	job.Request.DeferTextRestore = true
	job.Config.RestoreTextModels = false
	job, err := service.enqueue(job)
	require.NoError(t, err)
	waitImageBatchTerminal(t, service, job)
	service.mu.Lock()
	status, deferred := job.Status, job.TextRestoreDeferred
	service.mu.Unlock()
	require.Equal(t, "completed", status)
	require.False(t, deferred)
	require.Zero(t, countImageEvent(events, "restore_text"))
}

func TestImageBatchHistoryFailurePreservesRegisteredImageAndRestores(t *testing.T) {
	_, service, router, events, runner := newImageBatchTestService(t)
	runner.afterSave = func(index int) {
		if index == 1 {
			_ = os.Remove(filepath.Join(service.root, "jobs.json"))
			_ = os.Mkdir(filepath.Join(service.root, "jobs.json"), 0700)
		}
	}
	job := imageLifecycleJob(service, "")
	job.Request.DeferTextRestore = true
	job, err := service.enqueue(job)
	require.NoError(t, err)
	waitImageBatchTerminal(t, service, job)
	service.mu.Lock()
	status, deferred := job.Status, job.TextRestoreDeferred
	result := *job.Result
	errorMessage := job.Error
	service.mu.Unlock()
	require.Equal(t, "failed", status)
	require.False(t, deferred)
	require.Contains(t, errorMessage, "history could not be persisted")
	require.Equal(t, 1, countImageEvent(events, "restore_text"))
	response := imageTestHTTP(router, "GET", "/api/images/assets/"+result.AssetID, nil, "")
	require.Equal(t, 200, response.Code)
}

func TestImageGenerationSnapshotAndSessionFilterPreserveRecoveryMetadata(t *testing.T) {
	_, service, router, _, _ := newImageBatchTestService(t)
	service.workerStarted = true
	service.config.Profiles[0].PromptPrefix = "visual style"
	service.config.Profiles[0].NegativePrompt = "avoid artifacts"
	body := []byte(`{"request_id":"snapshot","session_id":"session-a","prompt":"raw prompt","negative_prompt":"raw negative","count":2,"width":96,"seed":8}`)
	response := imageTestHTTP(router, "POST", "/api/images/jobs", body, "application/json")
	require.Equal(t, 202, response.Code, response.Body.String())
	service.mu.Lock()
	summary := service.jobs[0].Generation
	service.jobs = append(service.jobs, &imageJob{ID: "other", Status: "completed", SessionID: "session-b"})
	service.mu.Unlock()
	require.Equal(t, "raw prompt", summary.UserPrompt)
	require.Equal(t, "visual style\nraw prompt", summary.Prompt)
	require.Equal(t, "avoid artifacts\nraw negative", summary.NegativePrompt)
	require.Equal(t, 96, summary.Profile.Width)
	require.Equal(t, 2, summary.Count)
	service.config.Profiles[0].PromptPrefix = "changed later"
	require.Equal(t, "visual style", summary.Profile.PromptPrefix)
	response = imageTestHTTP(router, "GET", "/api/images/jobs?session_id=session-a&request_id=snapshot", nil, "")
	var listed struct {
		Jobs []imageJob `json:"jobs"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &listed))
	require.Len(t, listed.Jobs, 1)
	require.Equal(t, "session-a", listed.Jobs[0].SessionID)
}

func TestImageBatchCountSeedAndCacheBoundsFailBeforeNativeWork(t *testing.T) {
	_, _, router, events, _ := newImageBatchTestService(t)
	for _, body := range []string{`{"prompt":"x","count":9}`, `{"prompt":"x","count":-1}`, `{"prompt":"x","count":2,"seed":9223372036854775807}`} {
		response := imageTestHTTP(router, "POST", "/api/images/jobs", []byte(body), "application/json")
		require.Equal(t, 400, response.Code, response.Body.String())
	}
	require.Empty(t, events.snapshot())
	config := defaultImageConfig(t.TempDir())
	require.Equal(t, 256, config.ConditioningCacheMiB)
	for _, limit := range []int{0, 8192} {
		config.ConditioningCacheMiB = limit
		require.NoError(t, validateImageConfig(config))
	}
	for _, limit := range []int{-1, 8193} {
		config.ConditioningCacheMiB = limit
		require.Error(t, validateImageConfig(config))
	}
}

func TestImageVariationLineageDoesNotSilentlyAddReferenceConditioning(t *testing.T) {
	_, service, router, _, runner := newImageBatchTestService(t)
	parent := imageLifecycleJob(service, "parent")
	require.NoError(t, service.generate(parent))
	child := imageLifecycleJob(service, "variation")
	child.Request.Count = 2
	child.Request.ParentAssetID = parent.Result.AssetID
	require.NoError(t, service.generate(child))
	require.Len(t, child.Results, 2)
	for _, result := range child.Results {
		require.Equal(t, parent.Result.AssetID, result.ParentAssetID)
		response := imageTestHTTP(router, "GET", "/api/images/assets/"+result.AssetID, nil, "")
		require.Equal(t, 200, response.Code)
	}
	require.Empty(t, child.Generation.ReferenceAssetIDs)
	runner.mu.Lock()
	calls := append([]imagegen.Request(nil), runner.calls...)
	runner.mu.Unlock()
	for _, request := range calls {
		require.Empty(t, request.ReferenceImages)
	}
}
