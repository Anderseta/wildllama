package server

import (
	"bytes"
	"context"
	"encoding/json"
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
	call := func(method, path string, body any) *httptest.ResponseRecorder {
		encoded, err := json.Marshal(body)
		require.NoError(t, err)
		return imageTestHTTP(router, method, path, encoded, "application/json")
	}
	saved := call("PUT", "/api/videos/config?mode=i2v", profile)
	require.Equal(t, 200, saved.Code, saved.Body.String())
	require.NotContains(t, call("GET", "/api/videos/config?mode=t2v", nil).Body.String(), "vision.gguf")
	input := map[string]any{"request_id": "test-wan", "prompt": "tilt up", "seed": int64(5232021495785965240)}
	require.Equal(t, 400, call("POST", "/api/videos/jobs", input).Code)
	upload := imageTestHTTP(router, "POST", imageUploadURI("first-frame", ""), imageUploadFixture(t, 150), "image/png")
	require.Equal(t, 201, upload.Code, upload.Body.String())
	reference, _ := imageUploadedResponse(t, upload)
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
