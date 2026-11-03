package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
)

var errImageResidentOnly = &api.StatusError{StatusCode: 503, Status: "503 Service Unavailable", ErrorMessage: "An image job is sharing the GPU. Only unchanged, already resident text models can run; retry after the image job finishes."}

type imageTextState struct {
	model                    *Model
	options                  api.Options
	duration                 time.Duration
	numCtxAuto, numBatchAuto bool
	useMMapAuto              bool
	shift                    bool
	key                      string
	pid                      int
}

type imageTextBackend interface {
	Suspend(context.Context) ([]imageTextState, error)
	Restore(context.Context, []imageTextState) []string
	FreeVRAM(context.Context) (uint64, error)
}

type schedulerImageText struct{ server *Server }

func (b *schedulerImageText) Suspend(ctx context.Context) ([]imageTextState, error) {
	if b.server.sched == nil {
		return nil, nil
	}
	scheduler := b.server.sched
	// Cancellation releases the HTTP reader before native Load/Close necessarily
	// finishes. Wait for construction, unpublished loaders and cleanup as well.
	if err := waitForImageTextLoaders(ctx, scheduler); err != nil {
		return nil, err
	}
	registry := scheduler.loadedSnapshot.Load()
	if registry == nil {
		return nil, nil
	}
	var saved []imageTextState
	for _, runner := range registry.runners {
		if !runner.waitForIdle(ctx, ctx) {
			return saved, ctx.Err()
		}
		if !runner.refMu.lockContext(ctx, ctx) {
			return saved, ctx.Err()
		}
		if runner.unloaded || runner.model.Load() == nil || runner.Options == nil {
			runner.refMu.Unlock()
			continue
		}
		var options api.Options
		data, err := json.Marshal(runner.Options)
		if err == nil {
			err = json.Unmarshal(data, &options)
		}
		if err != nil {
			runner.refMu.Unlock()
			return saved, err
		}
		saved = append(saved, imageTextState{model: runner.model.Load(), options: options, duration: runner.sessionDuration,
			numCtxAuto: runner.numCtxAuto, numBatchAuto: runner.numBatchAuto, useMMapAuto: runner.useMMapAuto, shift: runner.contextShift, key: runner.modelKey, pid: runner.pid})
		runner.sessionDuration = 0
		if runner.expireTimer != nil {
			runner.expireTimer.Stop()
			runner.expireTimer = nil
		}
		runner.publishStatusLocked()
		runner.refMu.Unlock()
		select {
		case scheduler.expiredCh <- runner:
		case <-ctx.Done():
			return saved, ctx.Err()
		}
	}
	// Ownership is the original runner's key+PID, never an arbitrary new process.
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		remaining := false
		current := scheduler.loadedSnapshot.Load()
		if current != nil {
			for _, state := range saved {
				if runner := current.byKey[state.key]; runner != nil && runner.pid == state.pid {
					remaining = true
					break
				}
			}
		}
		if !remaining {
			return saved, nil
		}
		select {
		case <-ctx.Done():
			return saved, ctx.Err()
		case <-ticker.C:
		}
	}
}

func waitForImageTextLoaders(ctx context.Context, scheduler *Scheduler) error {
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()
	for {
		registry := scheduler.loadedSnapshot.Load()
		loading := registry != nil && registry.loading != nil
		if registry != nil {
			for _, runner := range registry.runners {
				if status := runner.status.Load(); status != nil && status.loading {
					loading = true
					break
				}
			}
		}
		if scheduler.imageLoads.Load() == 0 && !loading {
			return ctx.Err()
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

func (b *schedulerImageText) Restore(ctx context.Context, states []imageTextState) []string {
	if b.server.sched == nil {
		return nil
	}
	var failures []string
	for _, state := range states {
		// An explicit keep_alive=0 means the original caller wanted no idle
		// residency. The next normal text request still loads it on demand.
		if state.duration == 0 {
			continue
		}
		loadCtx, cancel := context.WithCancel(ctx)
		keep := &api.Duration{Duration: state.duration}
		shift := state.shift
		success, failed := b.server.sched.getRunnerWithMmapOrigin(loadCtx, state.model, state.options, keep, state.numCtxAuto, state.numBatchAuto, &shift, state.useMMapAuto)
		select {
		case <-success:
		case err := <-failed:
			if err != nil {
				failures = append(failures, fmt.Sprintf("%s: %v", state.model.Name, err))
			}
		case <-loadCtx.Done():
			failures = append(failures, fmt.Sprintf("%s: %v", state.model.Name, loadCtx.Err()))
		}
		cancel()
		if ctx.Err() != nil {
			break
		}
	}
	return failures
}

func (b *schedulerImageText) FreeVRAM(ctx context.Context) (uint64, error) {
	if b.server.sched != nil {
		if err := waitForImageTextLoaders(ctx, b.server.sched); err != nil {
			return 0, err
		}
	}
	// Fresh driver observation is independent of text's per-runner quota. It
	// includes other programs and is not a reservation or process hard limit.
	devices := discover.GPUDevices(ctx, nil)
	var free uint64
	found := false
	for _, device := range devices {
		if !strings.EqualFold(device.Library, "CUDA") {
			continue
		}
		if !found || device.FreeMemory < free {
			free = device.FreeMemory
		}
		found = true
	}
	if !found {
		return 0, errors.New("no CUDA device with observable free VRAM is available")
	}
	return free, nil
}
