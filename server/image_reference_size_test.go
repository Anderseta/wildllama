package server

import (
	"bytes"
	"encoding/json"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestImageReferenceSizeConfigAndSnapshotPersist(t *testing.T) {
	_, service, router, events, _ := newImageBatchTestService(t)
	service.workerStarted = true // inspect admission and persistence without running an image
	response := imageTestHTTP(router, "GET", "/api/images/config", nil, "")
	require.Contains(t, response.Body.String(), `"preserve_reference_size":false`)
	config := cloneImageConfig(service.config)
	config.Profiles[0].PreserveReferenceSize = true
	encoded, err := json.Marshal(config)
	require.NoError(t, err)
	response = imageTestHTTP(router, "PATCH", "/api/images/config", encoded, "application/json")
	require.Equal(t, 200, response.Code, response.Body.String())
	stored, err := os.ReadFile(filepath.Join(service.root, "config.json"))
	require.NoError(t, err)
	var restored imageConfig
	require.NoError(t, json.Unmarshal(stored, &restored))
	require.True(t, restored.Profiles[0].PreserveReferenceSize)
	savedWidth, savedHeight := restored.Profiles[0].Width, restored.Profiles[0].Height

	var original bytes.Buffer
	require.NoError(t, png.Encode(&original, image.NewRGBA(image.Rect(0, 0, 1920, 1080))))
	path := filepath.Join(config.OutputDir, "inputs", "reference", "image.png")
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0700))
	require.NoError(t, os.WriteFile(path, original.Bytes(), 0600))
	asset := imageAssetRecord{imageAsset: imageAsset{AssetID: "reference/image.png", JobID: "reference", Filename: "image.png", MIMEType: "image/png", Width: 1920, Height: 1080}, OutputRoot: config.OutputDir, Input: true}
	service.assets[asset.AssetID] = asset
	body := []byte(`{"request_id":"reference-size","session_id":"chat","prompt":"synthetic size fixture","input_asset_id":"reference/image.png","width":31,"height":9999}`)
	response = imageTestHTTP(router, "POST", "/api/images/jobs", body, "application/json")
	require.Equal(t, 202, response.Code, response.Body.String())
	require.Len(t, service.jobs, 1)
	job := service.jobs[0]
	require.True(t, job.Generation.Profile.PreserveReferenceSize)
	require.Equal(t, 1920, job.Generation.Profile.Width)
	require.Equal(t, 1088, job.Generation.Profile.Height)
	require.Equal(t, savedWidth, service.config.Profiles[0].Width)
	require.Equal(t, savedHeight, service.config.Profiles[0].Height)
	require.NotContains(t, events.snapshot(), "run_image")
	require.NotContains(t, events.snapshot(), "suspend_text")

	// Retries return the original snapshot even after a profile is edited.
	service.config.Profiles[0].PreserveReferenceSize = false
	response = imageTestHTTP(router, "POST", "/api/images/jobs", body, "application/json")
	require.Equal(t, 202, response.Code, response.Body.String())
	require.Len(t, service.jobs, 1)
	require.True(t, service.jobs[0].Generation.Profile.PreserveReferenceSize)
	response = imageTestHTTP(router, "POST", "/api/images/jobs", bytes.Replace(body, []byte(`"width":31`), []byte(`"width":32`), 1), "application/json")
	require.Equal(t, 409, response.Code, response.Body.String())
	stored, err = os.ReadFile(filepath.Join(service.root, "jobs.json"))
	require.NoError(t, err)
	var jobs []imageJob
	require.NoError(t, json.Unmarshal(stored, &jobs))
	require.True(t, jobs[0].Generation.Profile.PreserveReferenceSize)
	require.Equal(t, 1088, jobs[0].Generation.Profile.Height)
	after, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, original.Bytes(), after)
}

func TestImageReferenceSizeNoReferenceAndDisabledKeepOverrides(t *testing.T) {
	profile := imageProfile{Profile: defaultImageConfig(t.TempDir()).Profiles[0].Profile}
	profile.PreserveReferenceSize = true
	generation, err := imageGenerationFor(imageJobRequest{Prompt: "fixture", Width: 128, Height: 160}, profile)
	require.NoError(t, err)
	require.Equal(t, 128, generation.Profile.Width)
	require.Equal(t, 160, generation.Profile.Height)
	profile.PreserveReferenceSize = false
	generation, err = imageGenerationFor(imageJobRequest{Prompt: "fixture", Width: 128, Height: 160, ReferenceImages: []string{"first", "second"}}, profile)
	require.NoError(t, err)
	require.Equal(t, 128, generation.Profile.Width)
	require.Equal(t, []string{"first", "second"}, generation.ReferenceAssetIDs)
	_, err = imageGenerationFor(imageJobRequest{Prompt: "fixture", Width: 31}, profile)
	require.Error(t, err)
}
