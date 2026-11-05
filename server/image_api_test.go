package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/imagegen"
	"github.com/stretchr/testify/require"
)

type imageTestEvents struct {
	mu     sync.Mutex
	values []string
}

func (e *imageTestEvents) add(value string) {
	e.mu.Lock()
	e.values = append(e.values, value)
	e.mu.Unlock()
}
func (e *imageTestEvents) snapshot() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.values...)
}

type imageTestRunner struct {
	events    *imageTestEvents
	request   imagegen.Request
	profile   imagegen.Profile
	entered   chan struct{}
	wait      bool
	release   <-chan struct{}
	status    *imagegen.RunnerStatus
	resultErr error
	afterSave func()
}

func (e *imageTestRunner) Status() imagegen.RunnerStatus {
	if e.status != nil {
		return *e.status
	}
	return imagegen.RunnerStatus{Ready: true, Revision: "test", RAMBudgetSupported: true}
}
func (e *imageTestRunner) Close() error { e.events.add("close_image"); return nil }
func (e *imageTestRunner) Run(ctx context.Context, profile imagegen.Profile, request imagegen.Request, progress func(imagegen.Progress)) (imagegen.Result, error) {
	e.request = request
	e.profile = profile
	e.events.add("run_image")
	if e.entered != nil {
		close(e.entered)
	}
	if e.wait {
		<-ctx.Done()
		return imagegen.Result{}, ctx.Err()
	}
	if e.release != nil {
		select {
		case <-e.release:
		case <-ctx.Done():
			return imagegen.Result{}, ctx.Err()
		}
	}
	progress(imagegen.Progress{Phase: "generation", Message: "Denoising", Current: 1, Total: 8})
	if err := os.MkdirAll(filepath.Dir(request.OutputPath), 0700); err != nil {
		return imagegen.Result{}, err
	}
	file, err := os.Create(request.OutputPath)
	if err != nil {
		return imagegen.Result{}, err
	}
	err = png.Encode(file, image.NewRGBA(image.Rect(0, 0, 64, 64)))
	closeErr := file.Close()
	if err != nil {
		return imagegen.Result{}, err
	}
	if closeErr != nil {
		return imagegen.Result{}, closeErr
	}
	if e.afterSave != nil {
		e.afterSave()
	}
	return imagegen.Result{Path: request.OutputPath, MIMEType: "image/png", Width: 64, Height: 64, Seed: request.Seed}, e.resultErr
}

type imageTestText struct {
	events   *imageTestEvents
	restored chan error
}

func (b *imageTestText) Suspend(ctx context.Context) ([]imageTextState, error) {
	b.events.add("suspend_text")
	options := api.DefaultOptions()
	options.NumCtx = 131072
	return []imageTextState{{model: &Model{Name: "original"}, options: options, duration: -1}}, ctx.Err()
}
func (b *imageTestText) Restore(ctx context.Context, states []imageTextState) []string {
	b.events.add("restore_text")
	if b.restored != nil {
		b.restored <- ctx.Err()
	}
	if len(states) != 1 || states[0].options.NumCtx != 131072 || states[0].duration != -1 {
		return []string{"original profile changed"}
	}
	return nil
}
func (b *imageTestText) FreeVRAM(context.Context) (uint64, error) {
	b.events.add("free_memory")
	return 16 << 30, nil
}

func newImageTestService(t *testing.T) (*Server, *imageService, *gin.Engine, *imageTestEvents, *imageTestRunner) {
	t.Helper()
	root := t.TempDir()
	server := &Server{}
	service := &imageService{server: server, root: root, config: defaultImageConfig(root), gate: newImageGPUGate(), queue: make(chan *imageJob, 16), assets: map[string]imageAssetRecord{}, cancelledRequests: map[string]time.Time{}, uploadSlot: make(chan struct{}, 1)}
	service.ctx, service.shutdownCancel = context.WithCancel(t.Context())
	service.config.ReserveVRAMMiB = 16
	profile := &service.config.Profiles[0]
	profile.Width, profile.Height, profile.MaxVRAMMiB = 64, 64, 64
	weights := filepath.Join(root, "fixture.gguf")
	require.NoError(t, os.WriteFile(weights, append([]byte("GGUF"), make([]byte, 20)...), 0600))
	profile.DiffusionModel, profile.TextEncoder, profile.VAE, profile.VisionEncoder = weights, weights, weights, weights
	events := &imageTestEvents{}
	runner := &imageTestRunner{events: events}
	service.newEngine = func(imagegen.Config) (imagegen.Runner, error) { return runner, nil }
	service.text = &imageTestText{events: events}
	server.imagesOnce.Do(func() { server.images = service })
	router := gin.New()
	server.registerImageRoutes(router)
	t.Cleanup(server.stopImageJobs)
	return server, service, router, events, runner
}

func imageTestHTTP(router http.Handler, method, path string, body []byte, kind string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, path, bytes.NewReader(body))
	if kind != "" {
		request.Header.Set("Content-Type", kind)
	}
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	return response
}

func TestImageExclusiveRestoresExactProfileAfterChildClose(t *testing.T) {
	_, service, _, events, runner := newImageTestService(t)
	seed := int64(0)
	profile := service.config.Profiles[0]
	profile.Seed = &seed
	job := &imageJob{ID: "fixture", Config: cloneImageConfig(service.config), Profile: profile, Request: imageJobRequest{Prompt: "Portuguese prompt: uma paisagem"}, ctx: service.ctx}
	require.NoError(t, service.generate(job))
	require.Equal(t, []string{"suspend_text", "free_memory", "run_image", "close_image", "free_memory", "restore_text"}, events.snapshot())
	require.Empty(t, job.RestoreErrors)
	require.Equal(t, int64(0), runner.request.Seed)
	require.NotNil(t, job.Result)
	response := imageTestHTTP(mustImageRouter(t, service.server), "GET", "/api/images/assets/fixture/image.png", nil, "")
	require.Equal(t, 200, response.Code)
	_, err := png.Decode(response.Body)
	require.NoError(t, err)
}

func TestImageWorkerStartTimestampRemainsStableAfterCompletion(t *testing.T) {
	_, service, router, _, runner := newImageTestService(t)
	runner.entered = make(chan struct{})
	release := make(chan struct{})
	runner.release = release
	job, err := service.enqueue(&imageJob{Kind: "generate", Config: cloneImageConfig(service.config), Profile: service.config.Profiles[0], Request: imageJobRequest{Prompt: "timestamp fixture"}})
	require.NoError(t, err)
	select {
	case <-runner.entered:
	case <-time.After(time.Second):
		t.Fatal("async worker did not enter the blocked image runner")
	}
	response := imageTestHTTP(router, "GET", "/api/images/jobs/"+job.ID, nil, "")
	require.Equal(t, 200, response.Code)
	var running struct {
		Job imageJob `json:"job"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &running))
	require.Equal(t, "running", running.Job.Status)
	require.NotNil(t, running.Job.StartedAt)
	require.Nil(t, running.Job.FinishedAt)
	observedStart := *running.Job.StartedAt
	// Completion is impossible until this external handoff, after the running
	// timestamp has already been observed through the public API.
	handoff := time.Now().UTC()
	close(release)
	require.Eventually(t, func() bool {
		service.mu.Lock()
		defer service.mu.Unlock()
		return imageTerminal(job.Status)
	}, 2*time.Second, time.Millisecond)
	response = imageTestHTTP(router, "GET", "/api/images/jobs/"+job.ID, nil, "")
	var completed struct {
		Job imageJob `json:"job"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &completed))
	require.Equal(t, "completed", completed.Job.Status, completed.Job.Error)
	require.NotNil(t, completed.Job.StartedAt)
	require.NotNil(t, completed.Job.FinishedAt)
	require.True(t, completed.Job.StartedAt.Equal(observedStart), "completion must preserve the previously observed start")
	require.True(t, completed.Job.FinishedAt.After(observedStart))
	require.False(t, completed.Job.FinishedAt.Before(handoff), "finish must follow the external release")
}

func mustImageRouter(t *testing.T, server *Server) *gin.Engine {
	t.Helper()
	r := gin.New()
	server.registerImageRoutes(r)
	return r
}

func TestImageCancelRestoresWithUncancelledCleanupContext(t *testing.T) {
	_, service, _, events, runner := newImageTestService(t)
	runner.wait = true
	runner.entered = make(chan struct{})
	restored := make(chan error, 1)
	service.text = &imageTestText{events: events, restored: restored}
	ctx, cancel := context.WithCancel(service.ctx)
	job := &imageJob{ID: "cancelled", Config: cloneImageConfig(service.config), Profile: service.config.Profiles[0], Request: imageJobRequest{Prompt: "test"}, ctx: ctx}
	done := make(chan error, 1)
	go func() { done <- service.generate(job) }()
	select {
	case <-runner.entered:
	case <-time.After(time.Second):
		t.Fatal("runner did not start")
	}
	cancel()
	require.ErrorIs(t, <-done, context.Canceled)
	require.NoError(t, <-restored)
	require.Equal(t, []string{"suspend_text", "free_memory", "run_image", "close_image", "free_memory", "restore_text"}, events.snapshot())
}

func TestImageShutdownCancelsCleanupRestore(t *testing.T) {
	server, service, _, events, runner := newImageTestService(t)
	runner.wait = true
	runner.entered = make(chan struct{})
	job := &imageJob{ID: "shutdown", Config: cloneImageConfig(service.config), Profile: service.config.Profiles[0], Request: imageJobRequest{Prompt: "test"}, ctx: service.ctx}
	done := make(chan error, 1)
	go func() { done <- service.generate(job) }()
	select {
	case <-runner.entered:
	case <-time.After(time.Second):
		t.Fatal("runner did not start")
	}
	server.stopImageJobs()
	require.ErrorIs(t, <-done, context.Canceled)
	require.NotContains(t, events.snapshot(), "restore_text")
}

func TestImageSimultaneousDoesNotUnloadText(t *testing.T) {
	_, service, _, events, _ := newImageTestService(t)
	job := &imageJob{ID: "shared", Config: cloneImageConfig(service.config), Profile: service.config.Profiles[0], Request: imageJobRequest{Prompt: "test", ExecutionMode: "simultaneous"}, ctx: service.ctx}
	require.NoError(t, service.generate(job))
	require.Equal(t, []string{"free_memory", "run_image", "close_image"}, events.snapshot())
	require.False(t, service.gate.shared.Load())
	job.Profile.MaxVRAMMiB = 0
	require.ErrorContains(t, service.generate(job), "positive image VRAM")
}

func TestImageInvalidComponentsDoNotUnloadText(t *testing.T) {
	_, service, _, events, _ := newImageTestService(t)
	profile := service.config.Profiles[0]
	profile.VAE = ""
	job := &imageJob{ID: "missing", Config: cloneImageConfig(service.config), Profile: profile, Request: imageJobRequest{Prompt: "test"}, ctx: service.ctx}
	require.ErrorContains(t, service.generate(job), "VAE")
	require.Empty(t, events.snapshot())
}

func TestImageRequestIdempotenceAndCancelBeforeSubmission(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	service.workerStarted = true // keep the bounded queue deterministic, without a worker
	response := imageTestHTTP(router, "POST", "/api/images/jobs/by-request/cancelled-request/cancel", nil, "")
	require.Equal(t, 200, response.Code)
	_, err := service.enqueue(&imageJob{RequestID: "cancelled-request", Fingerprint: "same"})
	require.ErrorContains(t, err, "cancelled before")
	first, err := service.enqueue(&imageJob{RequestID: "same-request", Fingerprint: "same"})
	require.NoError(t, err)
	second, err := service.enqueue(&imageJob{RequestID: "same-request", Fingerprint: "same"})
	require.NoError(t, err)
	require.Same(t, first, second)
	require.Len(t, service.queue, 1)
	_, err = service.enqueue(&imageJob{RequestID: "same-request", Fingerprint: "changed"})
	require.Error(t, err)
	require.Len(t, service.jobs, 1)
}

func TestImageConfigRevisionAndSeedSurviveRestart(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	config := cloneImageConfig(service.config)
	seed := int64(0)
	config.Profiles[0].Seed = &seed
	data, err := json.Marshal(config)
	require.NoError(t, err)
	response := imageTestHTTP(router, "PATCH", "/api/images/config", data, "application/json")
	require.Equal(t, 200, response.Code, response.Body.String())
	response = imageTestHTTP(router, "PATCH", "/api/images/config", data, "application/json")
	require.Equal(t, 409, response.Code)
	t.Setenv("WILDLLAMA_IMAGES_DIR", service.root)
	restarted := &Server{}
	t.Cleanup(restarted.stopImageJobs)
	loaded := restarted.imageService()
	require.Equal(t, config.Revision+1, loaded.config.Revision)
	require.NotNil(t, loaded.config.Profiles[0].Seed)
	require.Equal(t, int64(0), *loaded.config.Profiles[0].Seed)
	seed = 123
	require.Equal(t, int64(0), *loaded.config.Profiles[0].Seed)
}

func TestImageHistoryRecoversNullAndInterruptedEntries(t *testing.T) {
	root := t.TempDir()
	t.Setenv("WILDLLAMA_IMAGES_DIR", root)
	require.NoError(t, os.WriteFile(filepath.Join(root, "jobs.json"), []byte(`[null,{"id":"old","status":"running"}]`), 0600))
	require.NoError(t, os.WriteFile(filepath.Join(root, "assets.json"), []byte(`null`), 0600))
	server := &Server{}
	t.Cleanup(server.stopImageJobs)
	service := server.imageService()
	require.Len(t, service.jobs, 1)
	require.Equal(t, "interrupted", service.jobs[0].Status)
	require.NotNil(t, service.assets)
	_, err := os.Stat(filepath.Join(root, "config.json"))
	require.True(t, errors.Is(err, os.ErrNotExist), "GET initialization must not create configuration")
}

func TestImageProgressIncludesDownloadComponentsAndClearsForGeneration(t *testing.T) {
	_, service, _, _, _ := newImageTestService(t)
	job := &imageJob{}
	service.updateDownload(job, imagegen.DownloadProgress{Phase: "download", Role: "vae", Filename: "vae.safetensors", Completed: 12, Total: 100, FileIndex: 3, FileCount: 4})
	require.Equal(t, "vae", job.Progress.Role)
	require.Equal(t, 3, job.Progress.FileIndex)
	require.Equal(t, "vae.safetensors", job.Progress.Filename)
	service.update(job, "generation", "Denoising", 1, 8)
	require.Empty(t, job.Progress.Role)
	require.Empty(t, job.Progress.Filename)
	require.Zero(t, job.Progress.FileCount)
}

func TestImageUploadAndRegisteredPreview(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	var body bytes.Buffer
	fixture := image.NewRGBA(image.Rect(0, 0, 64, 32))
	fixture.Set(0, 0, color.White)
	require.NoError(t, png.Encode(&body, fixture))
	response := imageTestHTTP(router, "POST", "/api/images/assets", body.Bytes(), "image/png")
	require.Equal(t, 201, response.Code, response.Body.String())
	var saved struct {
		Asset imageAsset `json:"asset"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &saved))
	require.Equal(t, 64, saved.Asset.Width)
	require.Equal(t, 32, saved.Asset.Height)
	response = imageTestHTTP(router, "GET", "/api/images/assets/"+saved.Asset.AssetID, nil, "")
	require.Equal(t, 200, response.Code)
	require.Equal(t, body.Bytes(), response.Body.Bytes())
	response = imageTestHTTP(router, "GET", "/api/images/assets/not-registered/image.png", nil, "")
	require.Equal(t, 404, response.Code)
	// PNG dimensions are in the first chunk; a valid header is not complete data.
	response = imageTestHTTP(router, "POST", "/api/images/assets", body.Bytes()[:33], "image/png")
	require.Equal(t, 400, response.Code)
	response = imageTestHTTP(router, "POST", "/api/images/assets", body.Bytes(), "image/jpeg")
	require.Equal(t, 400, response.Code)
	require.Len(t, service.assets, 1)
	_, err := openImageAsset(imageAssetRecord{imageAsset: imageAsset{JobID: "..", Filename: "image.png", MIMEType: "image/png"}, OutputRoot: service.config.OutputDir})
	require.Error(t, err)
}

func TestImageInstallRecipeAliasDefaultsVisionWithoutDownloading(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	service.transferStarted = true // inspect the I/O queue without starting a transfer
	body := []byte(`{"recipe_id":"` + imagegen.DefaultRecipeID + `","request_id":"recipe-request"}`)
	response := imageTestHTTP(router, "POST", "/api/images/models/install", body, "application/json")
	require.Equal(t, 202, response.Code, response.Body.String())
	require.Len(t, service.jobs, 1)
	require.True(t, service.jobs[0].IncludeVision)
	require.Equal(t, imagegen.DefaultRecipeID, service.jobs[0].RecipeID)
	response = imageTestHTTP(router, "POST", "/api/images/models/install", []byte(`{"recipe_id":"one","pipeline_id":"other"}`), "application/json")
	require.Equal(t, 400, response.Code)
}
