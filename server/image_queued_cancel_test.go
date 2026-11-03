package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/imagegen"
	"github.com/stretchr/testify/require"
)

func TestImageQueuedCancelPublishesTerminalBeforeActiveJobFinishes(t *testing.T) {
	for _, byRequest := range []bool{false, true} {
		t.Run(map[bool]string{false: "job-id", true: "request-id"}[byRequest], func(t *testing.T) {
			_, service, router, events, runner := newImageBatchTestService(t)
			runner.blockAt, runner.blocked = 1, make(chan struct{})
			release := make(chan struct{})
			runner.release = release
			first, err := service.enqueue(imageLifecycleJob(service, ""))
			require.NoError(t, err)
			select {
			case <-runner.blocked:
			case <-time.After(time.Second):
				t.Fatal("first image did not begin")
			}
			queued := imageLifecycleJob(service, "")
			queued.RequestID = "cancel-queued-request"
			queued, err = service.enqueue(queued)
			require.NoError(t, err)
			url := "/api/images/jobs/" + queued.ID + "/cancel"
			if byRequest {
				url = "/api/images/jobs/by-request/" + queued.RequestID + "/cancel"
			}
			response := imageTestHTTP(router, "POST", url, nil, "")
			require.Equal(t, 200, response.Code, response.Body.String())
			var result struct {
				Job imageJob `json:"job"`
			}
			require.NoError(t, json.Unmarshal(response.Body.Bytes(), &result))
			require.Equal(t, "cancelled", result.Job.Status)
			require.NotNil(t, result.Job.FinishedAt)
			require.Nil(t, result.Job.StartedAt)
			data, err := os.ReadFile(filepath.Join(service.root, "jobs.json"))
			require.NoError(t, err)
			var saved []imageJob
			require.NoError(t, json.Unmarshal(data, &saved))
			require.Len(t, saved, 2)
			require.Equal(t, "running", saved[0].Status)
			require.Equal(t, "cancelled", saved[1].Status)
			require.Equal(t, result.Job.FinishedAt, saved[1].FinishedAt)
			following, err := service.enqueue(imageLifecycleJob(service, ""))
			require.NoError(t, err)
			close(release)
			waitImageBatchTerminal(t, service, first)
			waitImageBatchTerminal(t, service, following)
			runner.mu.Lock()
			calls := append([]imagegen.Request(nil), runner.calls...)
			runner.mu.Unlock()
			require.Len(t, calls, 2)
			require.Contains(t, calls[0].OutputPath, first.ID)
			require.Contains(t, calls[1].OutputPath, following.ID)
			require.Equal(t, 2, countImageEvent(events, "suspend_text"))
			require.Equal(t, 2, countImageEvent(events, "restore_text"))
			service.mu.Lock()
			status, finishedAt, startedAt := queued.Status, queued.FinishedAt, queued.StartedAt
			service.mu.Unlock()
			require.Equal(t, "cancelled", status)
			require.Equal(t, result.Job.FinishedAt, finishedAt)
			require.Nil(t, startedAt)
		})
	}
}

func TestImageQueuedCancelReclaimsFullQueueWithoutReorderingSurvivors(t *testing.T) {
	_, service, router, events, runner := newImageBatchTestService(t)
	runner.blockAt, runner.blocked = 1, make(chan struct{})
	release := make(chan struct{})
	runner.release = release
	first, err := service.enqueue(imageLifecycleJob(service, ""))
	require.NoError(t, err)
	select {
	case <-runner.blocked:
	case <-time.After(time.Second):
		t.Fatal("first image did not begin")
	}
	queued := make([]*imageJob, cap(service.queue))
	for i := range queued {
		queued[i], err = service.enqueue(imageLifecycleJob(service, ""))
		require.NoError(t, err)
	}
	_, err = service.enqueue(imageLifecycleJob(service, ""))
	require.ErrorIs(t, err, errImageGPUQueueFull)
	cancelled := len(queued) / 2
	response := imageTestHTTP(router, "POST", "/api/images/jobs/"+queued[cancelled].ID+"/cancel", nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	replacement, err := service.enqueue(imageLifecycleJob(service, ""))
	require.NoError(t, err, "canceling a queued request must reclaim capacity before the active image finishes")
	close(release)
	waitImageBatchTerminal(t, service, replacement)
	expected := []string{first.ID}
	for i, job := range queued {
		if i != cancelled {
			expected = append(expected, job.ID)
		}
	}
	expected = append(expected, replacement.ID)
	runner.mu.Lock()
	actual := make([]string, len(runner.calls))
	for i, call := range runner.calls {
		actual[i] = filepath.Base(filepath.Dir(call.OutputPath))
	}
	runner.mu.Unlock()
	require.Equal(t, expected, actual)
	require.Equal(t, len(expected), countImageEvent(events, "suspend_text"))
	require.Equal(t, len(expected), countImageEvent(events, "restore_text"))
}
