package server

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/require"
)

func TestImageHandoffPreservesAutomaticMmapWithoutReloadingNextTurn(t *testing.T) {
	for _, test := range []struct {
		name             string
		automatic, value bool
	}{{"automatic-enabled", true, true}, {"automatic-disabled", true, false}, {"explicit-disabled", false, false}} {
		t.Run(test.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			scheduler := InitScheduler(ctx)
			options := api.DefaultOptions()
			options.NumCtx = 32
			options.NumGPU = 0
			value := test.value
			options.UseMMap = &value
			model := &Model{Name: "mmap-handoff", ModelPath: "mmap-handoff"}
			original := withModel(&runnerRef{modelKey: model.ModelPath, modelPath: model.ModelPath, Options: &options, llama: &mockLlm{}, contextShift: true, useMMapAuto: test.automatic, sessionDuration: time.Hour}, model)
			scheduler.loaded[model.ModelPath] = original
			scheduler.publishLoadedSnapshotLocked()
			unloaded := make(chan struct{})
			go func() {
				defer close(unloaded)
				select {
				case runner := <-scheduler.expiredCh:
					scheduler.loadedMu.Lock()
					delete(scheduler.loaded, runner.modelKey)
					scheduler.publishLoadedSnapshotLocked()
					scheduler.loadedMu.Unlock()
				case <-ctx.Done():
				}
			}()
			backend := &schedulerImageText{server: &Server{sched: scheduler}}
			states, err := backend.Suspend(ctx)
			require.NoError(t, err)
			require.Len(t, states, 1)
			require.Equal(t, test.automatic, states[0].useMMapAuto)
			<-unloaded
			restored := make(chan []string, 1)
			go func() { restored <- backend.Restore(ctx, states) }()
			var request *LlmRequest
			select {
			case request = <-scheduler.pendingReqCh:
			case <-time.After(time.Second):
				t.Fatal("restoration was not scheduled")
			}
			scheduler.releasePendingSlot(request)
			require.Equal(t, test.automatic, request.useMMapAuto)
			runner := withModel(&runnerRef{modelKey: model.ModelPath, modelPath: model.ModelPath, Options: &request.opts, llama: &mockLlm{}, contextShift: true, useMMapAuto: request.useMMapAuto, sessionDuration: time.Hour}, model)
			scheduler.loadedMu.Lock()
			scheduler.loaded[model.ModelPath] = runner
			scheduler.publishLoadedSnapshotLocked()
			scheduler.loadedMu.Unlock()
			require.True(t, request.useLoadedRunner(runner, scheduler.finishedReqCh))
			require.Empty(t, <-restored)
			next := options
			if test.automatic {
				next.UseMMap = nil
			}
			success, failed := scheduler.getRunner(ctx, model, next, nil, false, false, nil)
			require.Empty(t, failed)
			require.Same(t, runner, <-success)
			require.Empty(t, scheduler.pendingReqCh, "the first text turn must reuse the restored runner")
			if !test.automatic {
				req := &LlmRequest{ctx: ctx, model: model, opts: next, contextShift: true}
				req.opts.UseMMap = nil
				runner.refMu.Lock()
				matches := runner.matchesRequestLocked(req)
				runner.refMu.Unlock()
				require.False(t, matches, "explicit mmap must not be misclassified as automatic")
			}
		})
	}
}
