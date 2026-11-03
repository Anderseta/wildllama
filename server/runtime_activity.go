package server

import (
	"net/http"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

type runnerActivityEntry struct {
	reporter         llm.RuntimeActivityReporter
	model            string
	pid              int
	context          int
	exclusive        bool
	loadingRequestID string
	modelRef         *atomic.Pointer[Model]
	statusRef        *atomic.Pointer[loadedModel]
}

type fixedRuntimeActivity llm.RuntimeActivity

func (a fixedRuntimeActivity) RuntimeActivity() llm.RuntimeActivity { return llm.RuntimeActivity(a) }

func newRunnerActivity(model *Model, runner llm.LlamaServer, options api.Options) *runnerActivityEntry {
	reporter, ok := runner.(llm.RuntimeActivityReporter)
	if !ok || model == nil {
		return nil
	}
	policy, _ := api.ResolveGPUExecution(options)
	return &runnerActivityEntry{reporter: reporter, model: model.ShortName, pid: runner.Pid(), context: options.NumCtx, exclusive: policy.Exclusive}
}

type runnerActivityResponse struct {
	Model         string              `json:"model"`
	PID           int                 `json:"pid"`
	ContextLength int                 `json:"context_length"`
	GPUExclusive  bool                `json:"gpu_exclusive"`
	Activity      llm.RuntimeActivity `json:"activity"`
}

func (s *Server) RuntimeActivityHandler(c *gin.Context) {
	entries := make([]*runnerActivityEntry, 0)
	queueLength, busy := 0, false
	if s.sched != nil {
		queueLength = len(s.sched.pendingReqCh) + int(s.sched.pendingBacklog.Load())
		// Loading can hold loadedMu while constructing a runner. A progress
		// endpoint must never wait for the operation it is meant to describe.
		if snapshot := s.sched.loadedSnapshot.Load(); snapshot != nil {
			entries = append(entries, snapshot.activities...)
			loading := snapshot.loading != nil
			for _, runner := range snapshot.runners {
				if status := runner.status.Load(); status != nil && status.loading {
					loading = true
					break
				}
			}
			if !loading {
				if preparing := s.sched.preparingActivity.Load(); preparing != nil {
					entries = append(entries, preparing)
				}
			}
		} else if s.sched.loadedMu.TryLock() {
			if s.sched.activeLoading != nil && s.sched.activeActivity != nil {
				entries = append(entries, s.sched.activeActivity)
			}
			for _, runner := range s.sched.loaded {
				if model := runner.model.Load(); runner.activity != nil && model != nil {
					entry := *runner.activity
					entry.model = model.ShortName
					entries = append(entries, &entry)
				}
			}
			s.sched.loadedMu.Unlock()
		} else {
			busy = true
		}
	}
	runners := make([]runnerActivityResponse, 0, len(entries))
	for _, entry := range entries {
		modelName := entry.model
		if entry.modelRef != nil {
			model := entry.modelRef.Load()
			if model == nil {
				continue
			}
			modelName = model.ShortName
		}
		activity := entry.reporter.RuntimeActivity()
		loading := entry.statusRef == nil
		if entry.statusRef != nil {
			if status := entry.statusRef.Load(); status != nil {
				loading = status.loading
			}
		}
		if activity.RequestID == "" && entry.loadingRequestID != "" && loading {
			activity.RequestID = entry.loadingRequestID
		}
		runners = append(runners, runnerActivityResponse{Model: modelName, PID: entry.pid, ContextLength: entry.context, GPUExclusive: entry.exclusive, Activity: activity})
	}
	c.Header("Cache-Control", "no-store")
	c.JSON(http.StatusOK, gin.H{"version": 1, "observed_at": time.Now().UnixMilli(), "scope": "shared_runner", "scheduler_busy": busy, "queue_length": queueLength, "admission": s.requestAdmission().snapshot(), "runners": runners})
}
