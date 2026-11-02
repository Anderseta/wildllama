package server

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
)

var errImageGPUQueueFull = errors.New("GPU image transition queue is full")

// A lease covers an entire HTTP inference stream, not just its first token.
// Image status/cancel/assets routes never take a lease. Writers are FIFO and
// stop new readers before draining existing requests, so repeated RAG/chat
// requests cannot indefinitely postpone an exclusive image job.
type imageGPUGate struct {
	mu                      sync.Mutex
	readers, waitingReaders int
	writer                  string
	sharedOwner             string
	shared                  atomic.Bool
	writers                 []string
	changed                 chan struct{}
	maxWaiting              int
	closed                  bool
}

func newImageGPUGate() *imageGPUGate {
	return &imageGPUGate{changed: make(chan struct{}), maxWaiting: 256}
}

func (g *imageGPUGate) notifyLocked() {
	if g.waitingReaders == 0 && len(g.writers) == 0 {
		return
	}
	close(g.changed)
	g.changed = make(chan struct{})
}

func (g *imageGPUGate) reader(ctx context.Context) (func(), error) {
	g.mu.Lock()
	waiting := false
	for {
		if g.closed {
			if waiting {
				g.waitingReaders--
			}
			g.mu.Unlock()
			return nil, context.Canceled
		}
		if err := ctx.Err(); err != nil {
			if waiting {
				g.waitingReaders--
			}
			g.mu.Unlock()
			return nil, err
		}
		if g.writer == "" && len(g.writers) == 0 {
			if waiting {
				g.waitingReaders--
			}
			g.readers++
			g.mu.Unlock()
			var once sync.Once
			return func() { once.Do(func() { g.mu.Lock(); g.readers--; g.notifyLocked(); g.mu.Unlock() }) }, nil
		}
		if !waiting {
			if g.waitingReaders >= g.maxWaiting {
				g.mu.Unlock()
				return nil, errImageGPUQueueFull
			}
			g.waitingReaders++
			waiting = true
		}
		changed := g.changed
		g.mu.Unlock()
		select {
		case <-ctx.Done():
		case <-changed:
		}
		g.mu.Lock()
	}
}

func (g *imageGPUGate) exclusive(ctx context.Context, owner string) (func(), error) {
	if owner == "" {
		return nil, errors.New("missing GPU lease owner")
	}
	g.mu.Lock()
	if g.closed {
		g.mu.Unlock()
		return nil, context.Canceled
	}
	if g.writer == owner {
		g.mu.Unlock()
		return nil, errors.New("GPU lease owner is already active")
	}
	for _, queued := range g.writers {
		if queued == owner {
			g.mu.Unlock()
			return nil, errors.New("GPU lease owner is already queued")
		}
	}
	g.writers = append(g.writers, owner)
	g.notifyLocked()
	for {
		if g.closed {
			g.mu.Unlock()
			return nil, context.Canceled
		}
		if err := ctx.Err(); err != nil {
			for i, queued := range g.writers {
				if queued == owner {
					g.writers = append(g.writers[:i], g.writers[i+1:]...)
					break
				}
			}
			g.notifyLocked()
			g.mu.Unlock()
			return nil, err
		}
		if g.writer == "" && g.sharedOwner == "" && g.readers == 0 && len(g.writers) > 0 && g.writers[0] == owner {
			g.writer = owner
			g.writers = g.writers[1:]
			g.mu.Unlock()
			var once sync.Once
			return func() {
				once.Do(func() {
					g.mu.Lock()
					if g.writer == owner {
						g.writer = ""
					}
					if g.sharedOwner == owner {
						g.sharedOwner = ""
						g.shared.Store(false)
					}
					g.notifyLocked()
					g.mu.Unlock()
				})
			}, nil
		}
		changed := g.changed
		g.mu.Unlock()
		select {
		case <-ctx.Done():
		case <-changed:
		}
		g.mu.Lock()
	}
}

func (g *imageGPUGate) shutdown() {
	g.mu.Lock()
	g.closed = true
	g.writer = ""
	g.sharedOwner = ""
	g.shared.Store(false)
	g.notifyLocked()
	g.mu.Unlock()
}

func (g *imageGPUGate) coexist(owner string) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.writer != owner || g.readers != 0 {
		return errors.New("image job does not own the drained GPU transition")
	}
	g.writer = ""
	g.sharedOwner = owner
	g.shared.Store(true)
	g.notifyLocked()
	return nil
}

func (s *Server) imageInferenceGate() gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodPost || !inferencePath(c.Request.URL.Path) {
			c.Next()
			return
		}
		started := time.Now()
		requestCtx, cancel := context.WithCancel(c.Request.Context())
		c.Request = c.Request.WithContext(requestCtx)
		release, err := s.imageService().gate.reader(requestCtx)
		if err != nil {
			requestErr := requestCtx.Err()
			cancel()
			if requestErr != nil {
				c.AbortWithStatus(499)
			} else {
				code := "image_gpu_unavailable"
				if errors.Is(err, errImageGPUQueueFull) {
					code = "image_gpu_queue_full"
				}
				c.Header("Retry-After", "1")
				c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{"error": err.Error(), "code": code, "retryable": true})
			}
			return
		}
		// Cancel scheduler work before announcing that this complete HTTP reader
		// has drained. A disconnected loader may still be releasing native memory.
		defer func() { cancel(); release() }()
		if elapsed := time.Since(started).Milliseconds(); elapsed > 0 {
			c.Header("X-Wildllama-GPU-Queue-Ms", strconv.FormatInt(elapsed, 10))
		}
		c.Next()
	}
}
