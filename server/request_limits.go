package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

type inferenceLimits struct {
	BodyBytes, MemoryBytes, DecodedBytes, OutputBytes int64
	InFlight, Queue, EmbedItems                       int
}

func positiveLimit(name string, fallback, maximum int64) int64 {
	value, err := strconv.ParseInt(os.Getenv(name), 10, 64)
	if err != nil || value <= 0 || value > maximum {
		return fallback
	}
	return value
}

func configuredInferenceLimits() inferenceLimits {
	return inferenceLimits{
		BodyBytes:    positiveLimit("WILDLLAMA_MAX_REQUEST_BYTES", 64<<20, 1<<30),
		MemoryBytes:  positiveLimit("WILDLLAMA_MAX_REQUEST_MEMORY_BYTES", 256<<20, 8<<30),
		InFlight:     int(positiveLimit("WILDLLAMA_MAX_IN_FLIGHT", 8, 1024)),
		Queue:        int(positiveLimit("WILDLLAMA_MAX_PENDING_REQUESTS", 128, 4096)),
		EmbedItems:   int(positiveLimit("WILDLLAMA_MAX_EMBEDDING_ITEMS", 4096, 65536)),
		DecodedBytes: positiveLimit("WILDLLAMA_MAX_EMBEDDING_MEDIA_BYTES", 48<<20, 1<<30),
		OutputBytes:  positiveLimit("WILDLLAMA_MAX_EMBEDDING_OUTPUT_BYTES", 32<<20, 1<<30),
	}
}

var errInferenceBusy = errors.New("inference queue is full; retry when capacity is available")
var errInferenceSize = errors.New("request exceeds the configured inference memory budget")

type admissionWaiter struct {
	weight      int64
	interactive bool
	ready       chan struct{}
	granted     bool
}

type inferenceAdmission struct {
	mu               sync.Mutex
	limits           inferenceLimits
	active           int
	bytes            int64
	waiting          []*admissionWaiter
	interactiveBurst int
}

type inferenceReservation struct {
	admission *inferenceAdmission
	waiter    *admissionWaiter
	released  bool
}

func (r *inferenceReservation) release() {
	a := r.admission
	a.mu.Lock()
	defer a.mu.Unlock()
	if r.released {
		return
	}
	r.released = true
	a.active--
	a.bytes -= r.waiter.weight
	a.grantLocked()
}

func (r *inferenceReservation) shrink(weight int64) {
	a := r.admission
	a.mu.Lock()
	defer a.mu.Unlock()
	if r.released || weight < 0 || weight >= r.waiter.weight {
		return
	}
	a.bytes -= r.waiter.weight - weight
	r.waiter.weight = weight
	a.grantLocked()
}

func (a *inferenceAdmission) snapshot() gin.H {
	if !a.mu.TryLock() {
		return gin.H{"busy": true}
	}
	defer a.mu.Unlock()
	return gin.H{"active": a.active, "queued": len(a.waiting), "reserved_bytes": a.bytes,
		"max_in_flight": a.limits.InFlight, "max_queue": a.limits.Queue, "memory_budget_bytes": a.limits.MemoryBytes}
}

func (s *Server) requestAdmission() *inferenceAdmission {
	s.admissionOnce.Do(func() { s.admission = &inferenceAdmission{limits: configuredInferenceLimits()} })
	return s.admission
}

func (a *inferenceAdmission) grantLocked() {
	for a.active < a.limits.InFlight {
		selected := -1
		preferInteractive := a.interactiveBurst < 4
		for pass := 0; pass < 2 && selected < 0; pass++ {
			for index, waiter := range a.waiting {
				if waiter.weight <= a.limits.MemoryBytes-a.bytes && waiter.interactive == preferInteractive {
					selected = index
					break
				}
			}
			preferInteractive = !preferInteractive
		}
		if selected < 0 {
			return
		}
		waiter := a.waiting[selected]
		a.waiting = append(a.waiting[:selected], a.waiting[selected+1:]...)
		waiter.granted = true
		a.active++
		a.bytes += waiter.weight
		if waiter.interactive {
			a.interactiveBurst++
		} else {
			a.interactiveBurst = 0
		}
		close(waiter.ready)
	}
}

func (a *inferenceAdmission) acquire(ctx context.Context, weight int64, interactive bool) (func(), error) {
	reservation, err := a.acquireReservation(ctx, weight, interactive)
	if err != nil {
		return nil, err
	}
	return reservation.release, nil
}

func (a *inferenceAdmission) acquireReservation(ctx context.Context, weight int64, interactive bool) (*inferenceReservation, error) {
	if weight > a.limits.MemoryBytes {
		return nil, errInferenceSize
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	waiter := &admissionWaiter{weight: weight, interactive: interactive, ready: make(chan struct{})}
	a.mu.Lock()
	if len(a.waiting) >= a.limits.Queue {
		a.mu.Unlock()
		return nil, errInferenceBusy
	}
	a.waiting = append(a.waiting, waiter)
	a.grantLocked()
	a.mu.Unlock()
	reservation := &inferenceReservation{admission: a, waiter: waiter}
	select {
	case <-waiter.ready:
		if err := ctx.Err(); err != nil {
			reservation.release()
			return nil, err
		}
		return reservation, nil
	case <-ctx.Done():
		a.mu.Lock()
		granted := waiter.granted
		if !granted {
			for index, item := range a.waiting {
				if item == waiter {
					a.waiting = append(a.waiting[:index], a.waiting[index+1:]...)
					break
				}
			}
			a.grantLocked()
		}
		a.mu.Unlock()
		if granted {
			reservation.release()
		}
		return nil, ctx.Err()
	}
}

func inferencePath(path string) bool {
	switch path {
	case "/api/chat", "/api/generate", "/api/embed", "/api/embeddings", "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses", "/v1/responses/compact", "/v1/messages", "/v1/systemone", "/v1/audio/transcriptions", "/v1/audio/speech":
		return true
	}
	return false
}

func requestID(value string) string {
	valid := len(value) > 0 && len(value) <= 128
	for _, char := range value {
		if !(char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z' || char >= '0' && char <= '9' || char == '-' || char == '_') {
			valid = false
			break
		}
	}
	if valid {
		return value
	}
	var bytes [16]byte
	if _, err := rand.Read(bytes[:]); err == nil {
		return hex.EncodeToString(bytes[:])
	}
	return strconv.FormatInt(time.Now().UnixNano(), 36)
}

func inferenceError(c *gin.Context, status int, code, message string) {
	c.AbortWithStatusJSON(status, gin.H{"error": message, "code": code, "retryable": status == 429 || status == 503, "request_id": c.GetString("wildllama_request_id")})
}

func (s *Server) inferenceGuard() gin.HandlerFunc {
	admission := s.requestAdmission()
	return func(c *gin.Context) {
		id := requestID(c.GetHeader("X-Request-ID"))
		c.Set("wildllama_request_id", id)
		c.Header("X-Request-ID", id)
		c.Request = c.Request.WithContext(api.WithRequestID(c.Request.Context(), id))
		if c.Request.Method != http.MethodPost || !inferencePath(c.Request.URL.Path) {
			c.Next()
			return
		}
		limits := admission.limits
		if c.Request.ContentLength > limits.BodyBytes {
			inferenceError(c, http.StatusRequestEntityTooLarge, "request_too_large", fmt.Sprintf("request body exceeds %d bytes", limits.BodyBytes))
			return
		}
		// Reserve before decoding JSON/base64. Unknown-length bodies reserve their
		// entire allowance; waiting requests hold no parsed prompts or media.
		size := c.Request.ContentLength
		extra := int64(1 << 20)
		if strings.HasSuffix(c.Request.URL.Path, "/embed") || strings.HasSuffix(c.Request.URL.Path, "/embeddings") {
			// Includes result vectors and their JSON encoding, before the input
			// shape/model dimensions are available. Ordinary RAG batches are tiny.
			extra += 3 * limits.OutputBytes
		}
		bodyLimit := limits.BodyBytes
		if size < 0 {
			// Chunked clients are valid too. Reserve/enforce the largest input
			// compatible with the aggregate budget instead of rejecting them all.
			bodyLimit = min(bodyLimit, max(0, (limits.MemoryBytes-extra)/3))
			size = bodyLimit
		}
		weight := size*3 + extra
		priority := c.GetHeader("X-Wildgoose-Priority")
		interactive := priority == "interactive" || priority != "background" && !strings.HasSuffix(c.Request.URL.Path, "/embed") && !strings.HasSuffix(c.Request.URL.Path, "/embeddings")
		started := time.Now()
		reservation, err := admission.acquireReservation(c.Request.Context(), weight, interactive)
		if err != nil {
			if errors.Is(err, errInferenceSize) {
				inferenceError(c, 413, "request_memory_limit", err.Error())
			} else if c.Request.Context().Err() == nil {
				c.Header("Retry-After", "1")
				inferenceError(c, 503, "inference_queue_full", err.Error())
			}
			return
		}
		defer reservation.release()
		c.Set("wildllama_reservation", reservation)
		c.Set("wildllama_input_reservation", size*3+(1<<20))
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, bodyLimit)
		c.Header("X-Wildllama-Queue-Ms", strconv.FormatInt(time.Since(started).Milliseconds(), 10))
		c.Next()
		slog.Debug("inference completed", "request_id", id, "path", c.Request.URL.Path, "status", c.Writer.Status(), "elapsed", time.Since(started))
	}
}

func bindInferenceJSON(c *gin.Context, value any) bool {
	err := c.ShouldBindJSON(value)
	if err == nil {
		return true
	}
	var size *http.MaxBytesError
	if errors.As(err, &size) {
		inferenceError(c, 413, "request_too_large", err.Error())
	} else if errors.Is(err, io.EOF) {
		inferenceError(c, 400, "invalid_request", "missing request body")
	} else {
		inferenceError(c, 400, "invalid_request", err.Error())
	}
	return false
}

func (s *Server) protocolCapabilities() gin.H {
	l := s.requestAdmission().limits
	return gin.H{"protocol_version": 1, "features": gin.H{
		"tool_call_ids": true, "request_id": true, "structured_errors": true, "prompt_progress": true,
		"request_admission": true, "embedding_input_limits": true, "multimodal_token_budget": true,
		"image_jobs": true, "image_profiles": true, "image_gpu_coordination": true, "image_local_import": true, "image_loras": true,
	}, "limits": gin.H{"request_body_bytes": l.BodyBytes, "max_in_flight": l.InFlight, "max_queue": l.Queue,
		"embedding_items": l.EmbedItems, "embedding_decoded_bytes": l.DecodedBytes, "embedding_output_bytes": l.OutputBytes}}
}
