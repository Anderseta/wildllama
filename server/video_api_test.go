package server

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/png"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/imagegen"
	"github.com/stretchr/testify/require"
)

type videoTestRunner struct {
	*imageTestRunner
	payload []byte
}

func (r *videoTestRunner) Status() imagegen.RunnerStatus {
	status := r.imageTestRunner.Status()
	status.WanVideoStagingSupported = true
	return status
}

func (r *videoTestRunner) Run(ctx context.Context, p imagegen.Profile, input imagegen.Request, progress func(imagegen.Progress)) (imagegen.Result, error) {
	r.request = input
	if err := os.MkdirAll(filepath.Dir(input.OutputPath), 0700); err != nil {
		return imagegen.Result{}, err
	}
	err := os.WriteFile(input.OutputPath, r.payload, 0600)
	return imagegen.Result{Path: input.OutputPath, MIMEType: "video/webm", Seed: input.Seed}, err
}
func TestWanPackageActivatesOnlyAfterSuccessAndPreservesEdits(t *testing.T) {
	_, s, _, _, _ := newImageTestService(t)
	p := defaultVideoProfile()
	job := &imageJob{ctx: context.Background(), Profile: p.runtimeProfile("i2v")}
	p.Steps = 12
	path := filepath.Join(s.root, "video-i2v.json")
	require.NoError(t, writeImageJSON(path, p))
	files := map[string]string{"diffusion": "wan.gguf", "text_encoder": "fp8.safetensors", "vae": "vae.safetensors", "vision_encoder": "clip.safetensors", "wan_vocabulary": "tokens.json"}
	require.NoError(t, s.applyWanVideoPackage(job, files))
	actual, err := s.loadVideoProfile("i2v")
	require.NoError(t, err)
	require.Equal(t, p.Steps, actual.Steps)
	require.Equal(t, files["wan_vocabulary"], actual.Vocabulary)
	actual.Encoder = "manual.gguf"
	require.NoError(t, writeImageJSON(path, actual))
	require.ErrorContains(t, s.applyWanVideoPackage(job, files), "editados")
	after, err := s.loadVideoProfile("i2v")
	require.NoError(t, err)
	require.Equal(t, actual, after)
	other, err := s.loadVideoProfile("t2v")
	require.NoError(t, err)
	require.Empty(t, other.Encoder)
}

func TestWanPackageInstallQueueIsSeparateAndIdempotent(t *testing.T) {
	_, s, router, _, _ := newImageTestService(t)
	s.transferStarted = true
	post := func(id string) *httptest.ResponseRecorder {
		return imageTestHTTP(router, "POST", "/api/videos/install", []byte(`{"request_id":"`+id+`"}`), "application/json")
	}
	require.Equal(t, 400, post("!").Code)
	require.Equal(t, 202, post("wan-test").Code)
	require.Equal(t, 202, post("wan-test").Code)
	require.Equal(t, 202, post("wan-another").Code)
	require.Len(t, s.jobs, 1)
	job := s.jobs[0]
	require.True(t, imageTransferJob(job))
	require.Contains(t, imageTestHTTP(router, "GET", "/api/videos/jobs", nil, "").Body.String(), job.ID)
	require.NotContains(t, imageTestHTTP(router, "GET", "/api/images/jobs", nil, "").Body.String(), job.ID)
	response := imageTestHTTP(router, "POST", "/api/videos/jobs/"+job.ID+"/cancel", nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	require.Equal(t, "cancelled", job.Status)
}

func TestWanVideoReferenceToStream(t *testing.T) {
	_, service, router, events, base := newImageTestService(t)
	data, err := os.ReadFile(filepath.Join("..", "imagegen", "testdata", "wan-five-frames.webm"))
	require.NoError(t, err)
	runner := &videoTestRunner{base, data}
	service.newEngine = func(imagegen.Config) (imagegen.Runner, error) { return runner, nil }
	profile := defaultVideoProfile()
	profile.Diffusion = service.config.Profiles[0].DiffusionModel
	profile.Encoder, profile.VAE = profile.Diffusion, profile.Diffusion
	profile.Vision = filepath.Join(service.root, "vision.gguf")
	weights, err := os.ReadFile(profile.Encoder)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(profile.Vision, weights, 0600))
	profile.Width, profile.Height, profile.Frames = 64, 64, 5
	hd := profile
	hd.Width, hd.Height = 1280, 720
	require.NoError(t, hd.validate("i2v"))
	hd.Height = 721
	require.Error(t, hd.validate("i2v"))
	call := func(method, path string, body any) *httptest.ResponseRecorder {
		encoded, err := json.Marshal(body)
		require.NoError(t, err)
		return imageTestHTTP(router, method, path, encoded, "application/json")
	}
	saved := call("PUT", "/api/videos/config?mode=i2v", profile)
	require.Equal(t, 200, saved.Code, saved.Body.String())
	estimate := profile
	estimate.Width, estimate.Height, estimate.Frames, estimate.FPS = 480, 832, 221, 22
	memory := call("POST", "/api/videos/memory?mode=i2v", estimate)
	require.Equal(t, 200, memory.Code, memory.Body.String())
	require.Contains(t, memory.Body.String(), `"rgb_float_bytes":1059102720`)
	require.Contains(t, memory.Body.String(), `"latent_frames":56`)
	require.NotContains(t, memory.Body.String(), "previous_run")
	estimate.FPS = 30
	second := call("POST", "/api/videos/memory?mode=i2v", estimate)
	require.Equal(t, 200, second.Code)
	for _, response := range []string{memory.Body.String(), second.Body.String()} {
		require.Contains(t, response, `"rgb_float_bytes":1059102720`)
		require.Contains(t, response, `"wan14b_qkv_bytes":5367398400`)
	}
	stored, err := service.loadVideoProfile("i2v")
	require.NoError(t, err)
	require.Equal(t, profile, stored)
	require.NotContains(t, call("GET", "/api/videos/config?mode=t2v", nil).Body.String(), "vision.gguf")
	input := map[string]any{"request_id": "test-wan", "prompt": "tilt up", "seed": int64(5232021495785965240)}
	require.Equal(t, 400, call("POST", "/api/videos/jobs", input).Code)
	upload := imageTestHTTP(router, "POST", imageUploadURI("first-frame", ""), imageUploadFixture(t, 150), "image/png")
	require.Equal(t, 201, upload.Code, upload.Body.String())
	reference, _ := imageUploadedResponse(t, upload)
	input["input_asset_id"] = reference.AssetID
	input["prepared_frame"] = true
	require.Equal(t, 409, call("POST", "/api/videos/jobs", input).Code)
	var fitted bytes.Buffer
	require.NoError(t, png.Encode(&fitted, image.NewRGBA(image.Rect(0, 0, 64, 64))))
	upload = imageTestHTTP(router, "POST", imageUploadURI("fitted-frame", ""), fitted.Bytes(), "image/png")
	require.Equal(t, 201, upload.Code)
	reference, _ = imageUploadedResponse(t, upload)
	input["input_asset_id"] = reference.AssetID
	response := call("POST", "/api/videos/jobs", input)
	require.Equal(t, 202, response.Code, response.Body.String())
	var result struct{ Job imageJob }
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &result))
	id := result.Job.ID
	require.Eventually(t, func() bool {
		var value struct{ Jobs []imageJob }
		resp := call("GET", "/api/videos/jobs", nil)
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &value))
		return len(value.Jobs) == 1 && value.Jobs[0].Status == "completed"
	}, 3*time.Second, 10*time.Millisecond)
	require.NotNil(t, runner.request.Video)
	require.Len(t, runner.request.ReferenceImages, 1)
	require.Contains(t, events.snapshot(), "restore_text")
	require.Contains(t, call("GET", "/api/videos/jobs", nil).Body.String(), `"seed":"5232021495785965240"`)
	require.NotContains(t, call("GET", "/api/images/jobs", nil).Body.String(), id)
	idem := call("POST", "/api/videos/jobs", input)
	require.Equal(t, 202, idem.Code)
	require.Contains(t, idem.Body.String(), id)
	input["prompt"] = "other"
	require.Equal(t, 409, call("POST", "/api/videos/jobs", input).Code)
	require.Equal(t, 200, call("HEAD", "/api/videos/files/"+id, nil).Code)
	req := httptest.NewRequest("GET", "/api/videos/files/"+id, nil)
	req.Header.Set("Range", "bytes=0-15")
	res := httptest.NewRecorder()
	router.ServeHTTP(res, req)
	require.Equal(t, 206, res.Code)
	require.True(t, bytes.Equal(data[:16], res.Body.Bytes()))
	require.Equal(t, "video/webm", res.Header().Get("Content-Type"))
}
