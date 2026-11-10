package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

func TestLowMemoryEmbeddingAdmissionUsesActualOutput(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	for _, unknown := range []bool{false, true} {
		for _, large := range []bool{false, true} {
			for _, endpoint := range []string{"/api/embed", "/v1/embeddings"} {
				name := endpoint + "/" + map[bool]string{false: "known", true: "chunked"}[unknown] + "/" + map[bool]string{false: "small", true: "output-too-large"}[large]
				t.Run(name, func(t *testing.T) {
					dimensions, count := 768, 2
					if large {
						dimensions, count = 32768, 200
					}
					writeShowManifestVariant(t, "embedding-budget", manifest.RunnerGGML, manifest.FormatGGUF, model.ConfigV2{
						ModelFormat: manifest.FormatGGUF, Capabilities: []string{string(model.CapabilityEmbedding)}, EmbedLen: dimensions,
					}, map[string]any{"general.architecture": "llama", "llama.context_length": uint32(64)})
					runner := &preparedEmbeddingFixture{}
					var loads atomic.Int32
					s := &Server{sched: &Scheduler{
						pendingReqCh: make(chan *LlmRequest, 1), finishedReqCh: make(chan *LlmRequest, 1), expiredCh: make(chan *runnerRef, 1),
						unloadedCh: make(chan any, 1), loaded: make(map[string]*runnerRef), getGpuFn: getGpuFn, getSystemInfoFn: getSystemInfoFn, waitForRecovery: 250 * time.Millisecond,
						loadFn: func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
							loads.Add(1)
							req.successCh <- &runnerRef{llama: runner}
							return false
						},
					}}
					go s.sched.Run(t.Context())
					limits := configuredInferenceLimits()
					limits.MemoryBytes = 64 << 20
					s.requestAdmission().limits = limits
					router := gin.New()
					router.Use(s.inferenceGuard())
					router.POST("/api/embed", s.EmbedHandler)
					router.POST("/v1/embeddings", cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), middleware.EmbeddingsMiddleware(), s.EmbedHandler)
					inputs := make([]string, count)
					for i := range inputs {
						inputs[i] = "one"
					}
					body, err := json.Marshal(api.EmbedRequest{Model: "embedding-budget", Input: inputs})
					if err != nil {
						t.Fatal(err)
					}
					request := httptest.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
					if endpoint == "/v1/embeddings" {
						request.Header.Set("Content-Encoding", "identity")
					}
					if unknown {
						request.ContentLength = -1
					}
					response := httptest.NewRecorder()
					router.ServeHTTP(response, request)
					wantStatus, wantLoads := http.StatusOK, int32(1)
					if large {
						wantStatus, wantLoads = http.StatusRequestEntityTooLarge, 0
					}
					if response.Code != wantStatus || loads.Load() != wantLoads {
						t.Fatalf("status=%d loads=%d want %d/%d body=%s", response.Code, loads.Load(), wantStatus, wantLoads, response.Body.String())
					}
					if !large {
						var result struct {
							Embeddings [][]float32 `json:"embeddings"`
							Data       []struct {
								Embedding []float32 `json:"embedding"`
							} `json:"data"`
						}
						if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
							t.Fatal(err)
						}
						if endpoint == "/v1/embeddings" {
							for _, item := range result.Data {
								result.Embeddings = append(result.Embeddings, item.Embedding)
							}
						}
						if len(result.Embeddings) != count {
							t.Fatalf("output count=%d", len(result.Embeddings))
						}
						for _, vector := range result.Embeddings {
							if len(vector) != 2 || vector[0] != float32(.6) || vector[1] != float32(.8) {
								t.Fatalf("embedding changed: %v", vector)
							}
						}
					}
					a := s.requestAdmission()
					a.mu.Lock()
					defer a.mu.Unlock()
					if a.active != 0 || a.bytes != 0 || len(a.waiting) != 0 {
						t.Fatalf("admission leaked after response: active=%d bytes=%d queued=%d", a.active, a.bytes, len(a.waiting))
					}
				})
			}
		}
	}
}

func TestLegacyEmbeddingKeepsConservativeOutputReservation(t *testing.T) {
	for _, unknown := range []bool{false, true} {
		t.Run(map[bool]string{false: "known", true: "chunked"}[unknown], func(t *testing.T) {
			s := &Server{}
			limits := configuredInferenceLimits()
			limits.MemoryBytes = 64 << 20
			s.requestAdmission().limits = limits
			router := gin.New()
			router.Use(s.inferenceGuard())
			called := false
			router.POST("/api/embeddings", func(*gin.Context) { called = true })
			request := httptest.NewRequest(http.MethodPost, "/api/embeddings", strings.NewReader(`{"model":"fixture","prompt":"small"}`))
			if unknown {
				request.ContentLength = -1
			}
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			if response.Code != http.StatusRequestEntityTooLarge || called || s.admission.active != 0 || s.admission.bytes != 0 {
				t.Fatalf("status=%d called=%v active=%d bytes=%d", response.Code, called, s.admission.active, s.admission.bytes)
			}
		})
	}
}

func TestEncodedEmbeddingKeepsConservativeAdmission(t *testing.T) {
	plain, err := json.Marshal(api.EmbedRequest{Model: "fixture", Input: strings.Repeat("x", 1<<20)})
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := zstd.NewWriter(nil, zstd.WithEncoderConcurrency(1))
	if err != nil {
		t.Fatal(err)
	}
	encoded := encoder.EncodeAll(plain, nil)
	encoder.Close()
	for _, memoryMiB := range []int64{64, 256} {
		t.Run(map[int64]string{64: "low-memory-keeps-rejection", 256: "chunked-decompression-keeps-reservation"}[memoryMiB], func(t *testing.T) {
			s := &Server{}
			limits := configuredInferenceLimits()
			limits.MemoryBytes = memoryMiB << 20
			s.requestAdmission().limits = limits
			router := gin.New()
			router.Use(s.inferenceGuard())
			called := false
			var inputReserved, reservedAfter int64
			router.POST("/v1/embeddings", func(c *gin.Context) {
				if _, exists := c.Get("wildllama_body_counter"); exists {
					t.Error("compressed wire bytes must not be used as the parsed input footprint")
				}
				inputReserved = c.MustGet("wildllama_input_reservation").(int64)
				c.Next()
			}, cloudPassthroughMiddleware(cloudErrRemoteInferenceUnavailable), middleware.EmbeddingsMiddleware(), func(c *gin.Context) {
				called = true
				var request api.EmbedRequest
				if !bindInferenceJSON(c, &request) {
					return
				}
				text, ok := request.Input.(string)
				if !ok || len(text) != 1<<20 {
					t.Fatal("real compatibility middleware did not decompress and rewrite the request")
				}
				if err := shrinkEmbeddingReservation(c, 768*4); err != nil {
					t.Fatalf("conservative encoded request was unexpectedly rejected: %v", err)
				}
				s.admission.mu.Lock()
				reservedAfter = s.admission.bytes
				s.admission.mu.Unlock()
				c.Status(http.StatusNoContent)
			})
			request := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(encoded))
			request.ContentLength = -1
			request.Header.Set("Content-Encoding", "zstd")
			response := httptest.NewRecorder()
			router.ServeHTTP(response, request)
			if memoryMiB == 64 {
				if response.Code != http.StatusRequestEntityTooLarge || called {
					t.Fatalf("encoded low-memory behavior changed: status=%d called=%v", response.Code, called)
				}
			} else {
				originalBodyLimit := min(limits.BodyBytes, (limits.MemoryBytes-(1<<20)-3*limits.OutputBytes)/3)
				if response.Code != http.StatusNoContent || !called || inputReserved != originalBodyLimit*3+(1<<20) || reservedAfter != inputReserved+3*768*4 || inputReserved <= int64(len(plain))*3 {
					t.Fatalf("status=%d called=%v input-reservation=%d retained=%d", response.Code, called, inputReserved, reservedAfter)
				}
			}
			if s.admission.active != 0 || s.admission.bytes != 0 || len(s.admission.waiting) != 0 {
				t.Fatal("encoded request leaked its original reservation")
			}
		})
	}
}

func TestEmbeddingReservationCannotGrowOrResurrectReleasedCapacity(t *testing.T) {
	limits := admissionLimits()
	limits.MemoryBytes = 64 << 20
	a := &inferenceAdmission{limits: limits}
	reservation, err := a.acquireReservation(t.Context(), 32<<20, false)
	if err != nil {
		t.Fatal(err)
	}
	if !reservation.shrink(32 << 20) {
		t.Fatal("an exact reservation must remain valid")
	}
	if reservation.shrink(33<<20) || a.bytes != 32<<20 {
		t.Fatal("output validation grew an existing reservation")
	}
	if !reservation.shrink(1<<20) || a.bytes != 1<<20 {
		t.Fatal("unused conservative capacity was not released")
	}
	reservation.release()
	if reservation.shrink(0) || a.active != 0 || a.bytes != 0 {
		t.Fatal("released capacity was changed or reacquired")
	}
}

func TestChunkedEmbeddingBodyStillHonorsMemoryBound(t *testing.T) {
	s := &Server{}
	limits := configuredInferenceLimits()
	limits.MemoryBytes = (1 << 20) + 3*64
	limits.BodyBytes = 1024
	s.requestAdmission().limits = limits
	router := gin.New()
	router.Use(s.inferenceGuard())
	parsed := false
	router.POST("/api/embed", func(c *gin.Context) {
		var request api.EmbedRequest
		parsed = bindInferenceJSON(c, &request)
	})
	request := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(`{"model":"fixture","input":"`+strings.Repeat("x", 100)+`"}`))
	request.ContentLength = -1
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusRequestEntityTooLarge || parsed || s.admission.active != 0 || s.admission.bytes != 0 {
		t.Fatalf("status=%d parsed=%v active=%d bytes=%d", response.Code, parsed, s.admission.active, s.admission.bytes)
	}
}

func TestLowMemoryEmbeddingCancellationDoesNotRetainAdmission(t *testing.T) {
	s := &Server{}
	limits := configuredInferenceLimits()
	limits.MemoryBytes, limits.InFlight = 64<<20, 1
	s.requestAdmission().limits = limits
	held, err := s.admission.acquireReservation(t.Context(), limits.MemoryBytes, false)
	if err != nil {
		t.Fatal(err)
	}
	defer held.release()
	router := gin.New()
	router.Use(s.inferenceGuard())
	called := false
	router.POST("/api/embed", func(*gin.Context) { called = true })
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	request := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(`{"model":"fixture","input":"small"}`)).WithContext(ctx)
	request.ContentLength = -1
	done := make(chan struct{})
	go func() { router.ServeHTTP(httptest.NewRecorder(), request); close(done) }()
	waitAdmissionQueue(t, s.admission, 1)
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("cancelled chunked embedding retained its queue wait")
	}
	held.release()
	if called || s.admission.active != 0 || s.admission.bytes != 0 || len(s.admission.waiting) != 0 {
		t.Fatalf("called=%v active=%d bytes=%d queued=%d", called, s.admission.active, s.admission.bytes, len(s.admission.waiting))
	}
}
