package server

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestImageSamplingCountersSurviveComponentTransitions(t *testing.T) {
	v := &imageService{}
	job := &imageJob{}
	v.setImageBatchProgress(job, 1, 2)
	v.update(job, "sampling", "", 2, 6)
	for _, phase := range []string{"loading_diffusion", "encoding", "decoding", "saving"} {
		v.update(job, phase, "", 128, 128)
		require.Equal(t, int64(2), job.Progress.SamplingCurrent)
		require.Equal(t, int64(6), job.Progress.SamplingTotal)
	}
	v.update(job, "starting", "retry", 0, 0)
	require.Zero(t, job.Progress.SamplingTotal)
	v.update(job, "sampling", "", 6, 6)
	v.setImageBatchProgress(job, 2, 2)
	require.Zero(t, job.Progress.SamplingTotal)
}

func TestImageJobSummaryIsBoundedAndDoesNotMutateHistory(t *testing.T) {
	_, service, router, _, _ := newImageBatchTestService(t)
	service.jobs = []*imageJob{
		{ID: "older", SessionID: "a", Status: "completed"},
		{ID: "other-chat", SessionID: "b", Status: "completed"},
		{ID: "latest", SessionID: "a", Status: "running", Generation: &imageGenerationSummary{Prompt: "preserve"}, Progress: imageProgress{SamplingCurrent: 2, SamplingTotal: 6}},
	}
	r := imageTestHTTP(router, "GET", "/api/images/jobs?summary=true&limit=1&session_id=a", nil, "")
	require.Equal(t, 200, r.Code)
	var result struct {
		Jobs []imageJob `json:"jobs"`
	}
	require.NoError(t, json.Unmarshal(r.Body.Bytes(), &result))
	require.Len(t, result.Jobs, 1)
	require.Equal(t, "latest", result.Jobs[0].ID)
	require.Nil(t, result.Jobs[0].Generation)
	require.Equal(t, int64(2), result.Jobs[0].Progress.SamplingCurrent)
	require.Equal(t, "preserve", service.jobs[2].Generation.Prompt)
	for _, invalid := range []string{"-1", "0", "201", "invalid"} {
		r = imageTestHTTP(router, "GET", "/api/images/jobs?limit="+invalid, nil, "")
		require.Equal(t, 400, r.Code)
	}
}
