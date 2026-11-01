package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

func admissionLimits() inferenceLimits {
	return inferenceLimits{InFlight: 1, Queue: 8, BodyBytes: 64, MemoryBytes: 8 << 20, EmbedItems: 2, DecodedBytes: 16, OutputBytes: 1 << 20}
}

func waitAdmissionQueue(t *testing.T, a *inferenceAdmission, count int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		a.mu.Lock()
		size := len(a.waiting)
		a.mu.Unlock()
		if size == count {
			return
		}
		runtime.Gosched()
	}
	t.Fatalf("admission queue did not reach %d", count)
}

func TestInferenceAdmissionPriorityAndCancellation(t *testing.T) {
	a := &inferenceAdmission{limits: admissionLimits()}
	release, err := a.acquire(t.Context(), 10, false)
	if err != nil {
		t.Fatal(err)
	}
	type result struct {
		name    string
		release func()
	}
	granted := make(chan result, 2)
	go func() {
		done, err := a.acquire(t.Context(), 10, false)
		if err == nil {
			granted <- result{"background", done}
		}
	}()
	waitAdmissionQueue(t, a, 1)
	go func() {
		done, err := a.acquire(t.Context(), 10, true)
		if err == nil {
			granted <- result{"interactive", done}
		}
	}()
	waitAdmissionQueue(t, a, 2)
	ctx, cancel := context.WithCancel(t.Context())
	cancelled := make(chan error, 1)
	go func() { _, err := a.acquire(ctx, 10, false); cancelled <- err }()
	waitAdmissionQueue(t, a, 3)
	cancel()
	if err := <-cancelled; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	waitAdmissionQueue(t, a, 2)
	release()
	first := <-granted
	if first.name != "interactive" {
		t.Fatalf("first = %s", first.name)
	}
	first.release()
	second := <-granted
	if second.name != "background" {
		t.Fatalf("second = %s", second.name)
	}
	second.release()
	second.release()
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.active != 0 || a.bytes != 0 || len(a.waiting) != 0 {
		t.Fatalf("leaked capacity: %+v", a)
	}
}

func TestInferenceAdmissionBoundsAndReleasesConcurrentRequests(t *testing.T) {
	l := admissionLimits()
	l.InFlight = 4
	l.Queue = 128
	a := &inferenceAdmission{limits: l}
	var active, maximum atomic.Int32
	var group sync.WaitGroup
	for range 100 {
		group.Go(func() {
			release, err := a.acquire(t.Context(), 64, false)
			if err != nil {
				t.Error(err)
				return
			}
			defer release()
			value := active.Add(1)
			for old := maximum.Load(); value > old; old = maximum.Load() {
				if maximum.CompareAndSwap(old, value) {
					break
				}
			}
			runtime.Gosched()
			active.Add(-1)
		})
	}
	group.Wait()
	if maximum.Load() > 4 || a.active != 0 || a.bytes != 0 {
		t.Fatalf("unbounded or leaked admission: max=%d active=%d bytes=%d", maximum.Load(), a.active, a.bytes)
	}
}

func TestEmbeddingReservationReleasesUnusedCapacity(t *testing.T) {
	l := admissionLimits()
	l.InFlight = 2
	l.MemoryBytes = 150
	a := &inferenceAdmission{limits: l}
	first, err := a.acquireReservation(t.Context(), 100, false)
	if err != nil {
		t.Fatal(err)
	}
	ready := make(chan *inferenceReservation, 1)
	go func() {
		next, err := a.acquireReservation(t.Context(), 100, false)
		if err == nil {
			ready <- next
		}
	}()
	waitAdmissionQueue(t, a, 1)
	first.shrink(10)
	select {
	case second := <-ready:
		second.release()
	case <-time.After(time.Second):
		t.Fatal("small batch retained maximum output reservation")
	}
	first.release()
	first.shrink(0)
	first.release()
	if a.active != 0 || a.bytes != 0 {
		t.Fatalf("reservation leak: active=%d bytes=%d", a.active, a.bytes)
	}
}

func TestInferenceAdmissionRejectsFullQueueAndOversizedReservation(t *testing.T) {
	l := admissionLimits()
	l.Queue = 1
	a := &inferenceAdmission{limits: l}
	release, _ := a.acquire(t.Context(), 10, false)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	waiting := make(chan error, 1)
	go func() { _, err := a.acquire(ctx, 10, false); waiting <- err }()
	waitAdmissionQueue(t, a, 1)
	if _, err := a.acquire(t.Context(), 10, true); !errors.Is(err, errInferenceBusy) {
		t.Fatal(err)
	}
	if _, err := a.acquire(t.Context(), l.MemoryBytes+1, true); !errors.Is(err, errInferenceSize) {
		t.Fatal(err)
	}
	cancel()
	<-waiting
	release()
}

func limitsRouter(handler gin.HandlerFunc) (*gin.Engine, *Server) {
	s := &Server{}
	s.requestAdmission().limits = admissionLimits()
	r := gin.New()
	r.Use(s.inferenceGuard())
	r.POST("/api/chat", handler)
	return r, s
}

func TestInferenceBodyLimitBeforeJSONAndUnknownLength(t *testing.T) {
	for _, unknown := range []bool{false, true} {
		t.Run(map[bool]string{false: "known", true: "chunked"}[unknown], func(t *testing.T) {
			called := false
			r, _ := limitsRouter(func(c *gin.Context) {
				var value any
				if bindInferenceJSON(c, &value) {
					called = true
				}
			})
			request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"text":"`+strings.Repeat("x", 100)+`"}`))
			if unknown {
				request.ContentLength = -1
			}
			request.Header.Set("X-Request-ID", "stable-request-123")
			response := httptest.NewRecorder()
			r.ServeHTTP(response, request)
			if response.Code != 413 || called {
				t.Fatalf("status=%d parsed=%t", response.Code, called)
			}
			if response.Header().Get("X-Request-ID") != "stable-request-123" {
				t.Fatal("request id was not preserved")
			}
			var payload map[string]any
			_ = json.Unmarshal(response.Body.Bytes(), &payload)
			if payload["code"] != "request_too_large" || payload["request_id"] != "stable-request-123" {
				t.Fatalf("error = %v", payload)
			}
		})
	}
}

func TestUnknownLengthEmbeddingFitsAggregateMemoryBudget(t *testing.T) {
	s := &Server{}
	s.requestAdmission().limits = admissionLimits()
	r := gin.New()
	r.Use(s.inferenceGuard())
	r.POST("/api/embed", func(c *gin.Context) { _, _ = io.Copy(io.Discard, c.Request.Body); c.Status(204) })
	request := httptest.NewRequest("POST", "/api/embed", strings.NewReader(`{}`))
	request.ContentLength = -1
	response := httptest.NewRecorder()
	r.ServeHTTP(response, request)
	if response.Code != 204 {
		t.Fatalf("chunked request rejected: %d %s", response.Code, response.Body.String())
	}
}

func TestEmbeddingInputLimitsRejectBeforeModelResolution(t *testing.T) {
	s := &Server{}
	s.requestAdmission().limits = admissionLimits()
	r := gin.New()
	r.Use(s.inferenceGuard())
	r.POST("/api/embed", s.EmbedHandler)
	request := httptest.NewRequest("POST", "/api/embed", strings.NewReader(`{"model":"absent","input":["a","a","a"]}`))
	response := httptest.NewRecorder()
	r.ServeHTTP(response, request)
	if response.Code != 413 || !strings.Contains(response.Body.String(), "embedding_input_limit") {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if _, err := validateEmbeddingInput(map[string]any{"image": strings.Repeat("YQ==", 40)}, admissionLimits()); err == nil {
		t.Fatal("decoded media budget ignored")
	}
}

func TestEmbeddingOutputUsesFullDimensionsBeforeTruncation(t *testing.T) {
	m := &Model{}
	m.Config.EmbedLen = 65536
	if err := validateEmbeddingOutput(100, m, api.EmbedRequest{Dimensions: 16}, admissionLimits()); err == nil {
		t.Fatal("raw output clones bypassed the budget")
	}
	m.Config.EmbedLen = 768
	if err := validateEmbeddingOutput(16, m, api.EmbedRequest{}, admissionLimits()); err != nil {
		t.Fatal(err)
	}
}

func TestProtocolCapabilitiesAndInvalidRequestID(t *testing.T) {
	s := &Server{}
	capabilities := s.protocolCapabilities()
	if capabilities["protocol_version"] != 1 {
		t.Fatal(capabilities)
	}
	features := capabilities["features"].(gin.H)
	if features["tool_call_ids"] != true || features["request_admission"] != true {
		t.Fatal(features)
	}
	for _, value := range []string{"bad\r\nheader", strings.Repeat("a", 129), ""} {
		id := requestID(value)
		if len(id) == 0 || len(id) > 128 || strings.ContainsAny(id, "\r\n") {
			t.Fatalf("unsafe id %q", id)
		}
	}
}
