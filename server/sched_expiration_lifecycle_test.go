package server

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/require"
)

func startExpirationFixture(t *testing.T, references uint, duration time.Duration) (*Scheduler, *runnerRef) {
	t.Helper()
	s, runner := idleFixture(t, references)
	runner.sessionDuration = duration
	s.finishedReqCh = make(chan *LlmRequest)
	s.expiredCh = make(chan *runnerRef, 1)
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.processCompleted(ctx)
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Error("completed-request loop did not stop")
		}
		runner.refMu.Lock()
		if runner.expireTimer != nil {
			runner.expireTimer.Stop()
		}
		runner.refMu.Unlock()
	})
	return s, runner
}

func enqueueExpirationFixture(t *testing.T, s *Scheduler, runner *runnerRef) {
	t.Helper()
	runner.refMu.Lock()
	s.expiredCh <- runner
	consumed := false
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if len(s.expiredCh) == 0 {
			consumed = true
			break
		}
		time.Sleep(time.Millisecond)
	}
	runner.refMu.Unlock()
	require.True(t, consumed, "scheduler did not receive the expiration event")
}

func finishExpirationFixture(t *testing.T, s *Scheduler, runner *runnerRef) {
	t.Helper()
	s.finishedReqCh <- &LlmRequest{model: runner.model.Load(), assigned: runner}
	require.Eventually(t, func() bool {
		return runner.status.Load() != nil
	}, time.Second, time.Millisecond)
}

func TestSchedulerStaleBusyExpirationPreservesKeepAlive(t *testing.T) {
	var forever api.Duration
	require.NoError(t, json.Unmarshal([]byte(`-1`), &forever))
	for _, test := range []struct {
		name     string
		duration time.Duration
	}{
		{"normal", time.Hour},
		{"infinite", forever.Duration},
	} {
		t.Run(test.name, func(t *testing.T) {
			s, runner := startExpirationFixture(t, 1, test.duration)
			enqueueExpirationFixture(t, s, runner)
			finishExpirationFixture(t, s, runner)
			select {
			case <-s.unloadedCh:
				t.Fatal("a busy expiration evicted a model after its keep-alive was renewed")
			case <-time.After(100 * time.Millisecond):
			}
			runner.refMu.Lock()
			defer runner.refMu.Unlock()
			require.False(t, runner.unloaded)
			require.NotNil(t, runner.expireTimer)
			require.Equal(t, test.duration, runner.sessionDuration)
		})
	}
}

func TestSchedulerStaleIdleExpirationPreservesNewTimer(t *testing.T) {
	s, runner := startExpirationFixture(t, 0, time.Hour)
	runner.refMu.Lock()
	runner.expireTimer = time.AfterFunc(time.Hour, func() {})
	runner.expiresAt = time.Now().Add(time.Hour)
	runner.refMu.Unlock()

	// A second request provides a processing barrier without changing this
	// runner's references or its newly renewed keep-alive timer.
	barrier := &runnerRef{modelKey: "barrier", modelPath: "barrier", refCount: 2}
	barrier.model.Store(&Model{ModelPath: barrier.modelPath})
	s.loadedMu.Lock()
	s.loaded[barrier.modelKey] = barrier
	s.loadedMu.Unlock()
	enqueueExpirationFixture(t, s, runner)
	finishExpirationFixture(t, s, barrier)
	runner.refMu.Lock()
	defer runner.refMu.Unlock()
	require.False(t, runner.unloaded, "a queued stale expiration bypassed the renewed keep-alive timer")
	require.NotNil(t, runner.expireTimer)
}

func TestSchedulerExplicitBusyExpirationStillUnloads(t *testing.T) {
	for _, oom := range []bool{false, true} {
		name := "explicit unload"
		if oom {
			name = "runtime OOM"
		}
		t.Run(name, func(t *testing.T) {
			s, runner := startExpirationFixture(t, 1, time.Hour)
			if oom {
				s.expireRunnersForRuntimeOOM(runner.model.Load(), context.DeadlineExceeded)
				// Only a real OOM should force expiration.
				runner.refMu.Lock()
				require.Equal(t, time.Hour, runner.sessionDuration)
				runner.refMu.Unlock()
				s.expireRunnersForRuntimeOOM(runner.model.Load(), errors.New("out of memory"))
			} else {
				s.expireRunner(runner.model.Load())
			}
			enqueueExpirationFixture(t, s, runner)
			s.finishedReqCh <- &LlmRequest{model: runner.model.Load(), assigned: runner}
			select {
			case <-s.unloadedCh:
			case <-time.After(time.Second):
				t.Fatal("explicit expiration did not unload after the active request finished")
			}
		})
	}
}

func TestSchedulerObsoleteTimerPreservesReplacement(t *testing.T) {
	s := &Scheduler{expiredCh: make(chan *runnerRef, 1)}
	old := time.NewTimer(time.Hour)
	defer old.Stop()
	replacement := time.NewTimer(time.Hour)
	defer replacement.Stop()
	runner := &runnerRef{expireTimer: replacement}
	s.expirationTimerFired(t.Context(), runner, &runnerExpiration{timer: old})
	require.Same(t, replacement, runner.expireTimer)
	require.True(t, replacement.Stop(), "the obsolete callback stopped a newer timer")
	require.Empty(t, s.expiredCh)
}

func TestSchedulerTimerDispatchReleasesLockAndStopsOnShutdown(t *testing.T) {
	s := &Scheduler{expiredCh: make(chan *runnerRef)}
	timer := time.NewTimer(time.Hour)
	defer timer.Stop()
	runner := &runnerRef{expireTimer: timer}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.expirationTimerFired(ctx, runner, &runnerExpiration{timer: timer})
	}()
	require.Eventually(t, func() bool {
		if !runner.refMu.TryLock() {
			return false
		}
		defer runner.refMu.Unlock()
		return runner.expireTimer == nil
	}, time.Second, time.Millisecond, "the timer callback kept refMu locked while its queue was blocked")
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("shutdown left the timer callback blocked on the expiration queue")
	}
}

func TestSchedulerNanosecondTimerPublication(t *testing.T) {
	s := &Scheduler{expiredCh: make(chan *runnerRef, 1)}
	for range 256 {
		runner := &runnerRef{sessionDuration: time.Nanosecond}
		runner.refMu.Lock()
		s.scheduleExpirationLocked(t.Context(), runner)
		runner.refMu.Unlock()
		select {
		case got := <-s.expiredCh:
			require.Same(t, runner, got)
		case <-time.After(time.Second):
			t.Fatal("the shortest positive timer did not expire")
		}
	}
}

func TestSchedulerImmediateExpirationDoesNotBlockOnFullQueue(t *testing.T) {
	s, runner := startExpirationFixture(t, 1, 0)
	blocker := &runnerRef{refCount: 1}
	t.Cleanup(func() {
		// Release a legacy self-send if the regression fails, so shutdown can
		// still join the scheduler instead of leaving a stuck test goroutine.
		select {
		case <-s.expiredCh:
		default:
		}
	})
	runner.refMu.Lock()
	s.finishedReqCh <- &LlmRequest{model: runner.model.Load(), assigned: runner}
	// The completion loop has received the request and is waiting for refMu.
	// Fill its expiration queue before it can process the final reference.
	s.expiredCh <- blocker
	runner.refMu.Unlock()
	select {
	case <-s.unloadedCh:
	case <-time.After(250 * time.Millisecond):
		t.Fatal("keep_alive=0 completion deadlocked on its own full expiration queue")
	}
	runner.refMu.Lock()
	defer runner.refMu.Unlock()
	require.True(t, runner.unloaded)
}

func TestQueuedHTTPAdmissionReturnsOnDisconnect(t *testing.T) {
	s, _ := idleFixture(t, 0)
	server := &Server{sched: s}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	result := make(chan error, 1)
	go func() {
		_, _, _, err := server.scheduleRunnerForModel(ctx, &Model{ModelPath: "queued"}, nil, nil, nil, nil)
		result <- err
	}()
	select {
	case queued := <-s.pendingReqCh:
		defer func() { queued.errCh <- context.Canceled }()
	case <-time.After(time.Second):
		t.Fatal("request did not reach admission")
	}
	cancel()
	select {
	case err := <-result:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("disconnected HTTP request still waits for a discarded queue entry")
	}
}

func TestSchedulerExplicitExpirationReleasesLockBeforeQueueWait(t *testing.T) {
	for _, oom := range []bool{false, true} {
		s, runner := idleFixture(t, 0)
		s.expiredCh = make(chan *runnerRef)
		done := make(chan struct{})
		go func() {
			defer close(done)
			if oom {
				s.expireRunnersForRuntimeOOM(runner.model.Load(), errors.New("out of memory"))
			} else {
				s.expireRunner(runner.model.Load())
			}
		}()
		observed := assertUnlockedAfterExpiration(runner)
		require.Same(t, runner, <-s.expiredCh)
		<-done
		require.True(t, observed, "expiration held refMu while waiting for the queue")
	}
}

func assertUnlockedAfterExpiration(r *runnerRef) bool {
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if r.refMu.TryLock() {
			expired := r.sessionDuration == 0
			r.refMu.Unlock()
			if expired {
				return true
			}
		}
		time.Sleep(time.Millisecond)
	}
	return false
}

func TestSchedulerUnrelatedWakeupCannotCompleteEviction(t *testing.T) {
	s, runner := idleFixture(t, 0)
	s.unloadedCh = make(chan any, 1)
	s.unloadedCh <- struct{}{} // a different model finished earlier
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Millisecond)
	defer cancel()
	require.False(t, s.waitForUnload(ctx, ctx, runner))
	// A full notification queue must still allow cleanup and further admissions.
	s.unloadedCh <- struct{}{}
	done := make(chan struct{})
	go func() { defer close(done); s.unloadExpiredRunner(runner) }()
	select {
	case <-done:
	case <-time.After(time.Second):
		<-s.unloadedCh
		t.Fatal("full wakeup queue stopped the completed-request loop")
	}
	require.True(t, s.waitForUnload(t.Context(), t.Context(), runner))
}

func TestSchedulerStaleEventsPreserveReplacementWithSamePID(t *testing.T) {
	s, old := startExpirationFixture(t, 0, 0)
	current := &runnerRef{modelKey: old.modelKey, pid: old.pid, refCount: 2}
	current.model.Store(old.model.Load())
	s.loadedMu.Lock()
	s.loaded[current.modelKey] = current
	s.loadedMu.Unlock()
	s.unloadExpiredRunner(old)
	s.finishedReqCh <- &LlmRequest{model: current.model.Load(), assigned: old}
	finishExpirationFixture(t, s, current)
	s.loadedMu.Lock()
	require.Same(t, current, s.loaded[current.modelKey])
	s.loadedMu.Unlock()
	current.refMu.Lock()
	defer current.refMu.Unlock()
	require.Equal(t, uint(1), current.refCount)
}
