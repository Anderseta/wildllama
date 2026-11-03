package server

import (
	"context"
	"reflect"
	"slices"
	"time"
)

const (
	pendingInteractiveBurst  = 4
	pendingAgingTime         = 30 * time.Second
	pendingGroupingOvertakes = 1
)

// Only the pending scheduler goroutine owns this bounded reorder window. Native
// loading is not preempted and each request retains its own context/result pair.
type pendingModelQueue struct {
	requests         []*LlmRequest
	interactiveBurst int
	groupedOvertakes int
	previous         *LlmRequest
}

// Channel slots alone are insufficient once requests have entered the reorder
// window. Count both locations against the original configured queue bound.
func (s *Scheduler) reservePendingSlot(req *LlmRequest) bool {
	limit := int64(cap(s.pendingReqCh))
	if limit == 0 {
		return true
	} // preserve an unbuffered scheduler's handoff
	for {
		count := s.pendingOwned.Load()
		if count >= limit {
			return false
		}
		if s.pendingOwned.CompareAndSwap(count, count+1) {
			req.queueOwned.Store(true)
			return true
		}
	}
}

func (s *Scheduler) releasePendingSlot(req *LlmRequest) {
	if req == nil {
		return
	}
	if req.queueOwned.CompareAndSwap(true, false) {
		s.pendingOwned.Add(-1)
	}
}

func (s *Scheduler) rejectPendingRequest(req *LlmRequest, err error) {
	if req == nil {
		return
	}
	s.releasePendingSlot(req)
	if req.ctx.Err() != nil {
		return
	} // the original caller already owns cancellation
	select {
	case req.errCh <- err:
	default:
	}
}

func (s *Scheduler) discardPendingQueue(queue *pendingModelQueue) {
	for _, req := range queue.requests {
		s.rejectPendingRequest(req, context.Canceled)
	}
	queue.requests = nil
	s.pendingBacklog.Store(0)
	// The listener/request contexts shut down with the scheduler. Release any
	// arrivals already accepted while the active request was being cancelled.
	for {
		select {
		case req, open := <-s.pendingReqCh:
			if !open {
				return
			}
			s.rejectPendingRequest(req, context.Canceled)
		default:
			return
		}
	}
}

func (s *Scheduler) nextPendingRequest(ctx context.Context, queue *pendingModelQueue) *LlmRequest {
	for {
		if ctx.Err() != nil {
			return nil
		}
		if len(queue.requests) == 0 {
			select {
			case <-ctx.Done():
				return nil
			case req, open := <-s.pendingReqCh:
				if !open {
					return nil
				}
				queue.add(req, time.Now())
			case <-s.unloadedCh:
				continue
			}
		}
		// Snapshot the available arrivals; a busy producer must not hold this
		// loop indefinitely. Production admission bounds window+channel together.
		arrivals := len(s.pendingReqCh)
		for range arrivals {
			select {
			case req, open := <-s.pendingReqCh:
				if !open {
					return nil
				}
				queue.add(req, time.Now())
			default:
			}
		}
		kept := queue.requests[:0]
		for _, req := range queue.requests {
			if req.ctx.Err() != nil {
				s.releasePendingSlot(req)
			} else {
				kept = append(kept, req)
			}
		}
		for i := len(kept); i < len(queue.requests); i++ {
			queue.requests[i] = nil
		}
		queue.requests = kept
		if len(kept) == 0 {
			s.pendingBacklog.Store(0)
			continue
		}
		pending := queue.take(time.Now(), s.loadedSnapshot.Load())
		s.pendingBacklog.Store(int64(len(queue.requests)))
		s.releasePendingSlot(pending)
		return pending
	}
}

func (queue *pendingModelQueue) add(req *LlmRequest, now time.Time) {
	if req.queuedAt.IsZero() {
		req.queuedAt = now
	}
	queue.requests = append(queue.requests, req)
}

func (queue *pendingModelQueue) take(now time.Time, resident *loadedRunnerSnapshot) *LlmRequest {
	interactive, background, agedBackground := -1, -1, false
	for i, req := range queue.requests {
		if req.background {
			if background < 0 {
				background = i
				agedBackground = now.Sub(req.queuedAt) >= pendingAgingTime
			}
		} else if interactive < 0 {
			interactive = i
		}
	}
	selected := interactive
	if background >= 0 && (interactive < 0 || queue.interactiveBurst >= pendingInteractiveBurst || agedBackground && queue.interactiveBurst > 0) {
		selected = background
	}
	oldest := queue.requests[selected]
	// At most one compatible request can pass the oldest same-priority item,
	// only while the previous model is still resident and the oldest is fresh.
	// A queued configuration change for that same model is an ordering fence.
	if queue.previous != nil && queue.groupedOvertakes < pendingGroupingOvertakes && now.Sub(oldest.queuedAt) < pendingAgingTime && resident != nil && resident.byKey[schedulerModelKey(queue.previous.model)] != nil {
		for i := selected; i < len(queue.requests); i++ {
			candidate := queue.requests[i]
			if candidate.background != oldest.background {
				continue
			}
			if schedulerModelKey(candidate.model) == schedulerModelKey(queue.previous.model) {
				if compatiblePendingModels(queue.previous, candidate) {
					selected = i
				}
				break
			}
		}
	}
	if queue.requests[selected] != oldest {
		queue.groupedOvertakes++
	} else {
		queue.groupedOvertakes = 0
	}
	req := queue.requests[selected]
	copy(queue.requests[selected:], queue.requests[selected+1:])
	queue.requests[len(queue.requests)-1] = nil
	queue.requests = queue.requests[:len(queue.requests)-1]
	if req.background {
		queue.interactiveBurst = 0
	} else {
		queue.interactiveBurst = min(pendingInteractiveBurst, queue.interactiveBurst+1)
	}
	queue.previous = req
	return req
}

func compatiblePendingModels(first, next *LlmRequest) bool {
	if first.model == nil || next.model == nil || schedulerModelKey(first.model) != schedulerModelKey(next.model) {
		return false
	}
	a, b := first.opts.Runner, next.opts.Runner
	if first.numCtxAuto && next.numCtxAuto {
		b.NumCtx = a.NumCtx
	}
	if first.numBatchAuto && next.numBatchAuto {
		b.NumBatch = a.NumBatch
	}
	if first.useMMapAuto && b.UseMMap == nil {
		b.UseMMap = a.UseMMap
	}
	if a.NumGPU < 0 && b.NumGPU < 0 {
		a.NumGPU = -1
		b.NumGPU = -1
	}
	return first.contextShift == next.contextShift &&
		first.model.Runner == next.model.Runner &&
		reflect.DeepEqual(a, b) &&
		slices.Equal(first.model.AdapterPaths, next.model.AdapterPaths) &&
		slices.Equal(first.model.ProjectorPaths, next.model.ProjectorPaths) &&
		first.model.DraftPath == next.model.DraftPath &&
		slices.Equal(first.model.DraftShardPaths, next.model.DraftShardPaths)
}
