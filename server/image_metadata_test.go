package server

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestImageMetadataSurvivesJobPruningAndProfileChanges(t *testing.T) {
	_, service, router, _, _ := newImageBatchTestService(t)
	job := imageLifecycleJob(service, "metadata")
	job.SessionID = "chat-a"
	job.Request.Prompt = "<image1> lake\nline 2"
	job.Request.NegativePrompt = "noise"
	job.Profile.PromptPrefix = "style"
	job.Profile.NegativePrompt = "border"
	require.NoError(t, service.generate(job))
	require.NotNil(t, job.Result)
	want := imageMetadataFor(job.Generation)
	require.Equal(t, "style\n<image1> lake\nline 2", want.Prompt)
	require.Equal(t, "border\nnoise", want.NegativePrompt)
	service.mu.Lock()
	service.jobs = []*imageJob{job}
	job.Status = "completed"
	for i := range 200 {
		service.jobs = append(service.jobs, &imageJob{ID: fmt.Sprint(i), Status: "completed"})
	}
	require.NoError(t, service.persistJobsLocked())
	require.Len(t, service.jobs, 200)
	job.Generation.Prompt = "later mutation"
	service.mu.Unlock()
	loaded := &imageService{root: service.root}
	require.NoError(t, loaded.loadImageCatalog())
	require.Equal(t, want, loaded.assets[job.Result.AssetID].Generation)
	response := imageTestHTTP(router, "GET", "/api/images/assets/"+job.Result.AssetID+"/metadata?session_id=chat-a", nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	var metadata struct {
		Generation *imageGenerationMetadata `json:"generation"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &metadata))
	require.Equal(t, want, metadata.Generation)
	response = imageTestHTTP(router, "GET", "/api/images/assets/"+job.Result.AssetID+"/metadata?session_id=chat-b", nil, "")
	require.Equal(t, 404, response.Code)
}

func TestImageMetadataBackfillUsesOnlyHistoricalSnapshot(t *testing.T) {
	service := &imageService{root: t.TempDir(), assets: map[string]imageAssetRecord{
		"old/image.png":     {imageAsset: imageAsset{AssetID: "old/image.png", JobID: "old"}},
		"unknown/image.png": {imageAsset: imageAsset{AssetID: "unknown/image.png", JobID: "unknown"}},
	}, jobs: []*imageJob{{ID: "old", Generation: &imageGenerationSummary{Prompt: "old prompt", NegativePrompt: "old negative"}}}}
	require.NoError(t, service.backfillImageMetadata())
	require.NoError(t, service.backfillImageMetadata())
	loaded := &imageService{root: service.root}
	require.NoError(t, loaded.loadImageCatalog())
	require.Equal(t, "old prompt", loaded.assets["old/image.png"].Generation.Prompt)
	require.Nil(t, loaded.assets["unknown/image.png"].Generation)
}
