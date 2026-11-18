package server

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/middleware"
)

func TestCompletionOnlyTokenizesReturnedContext(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_GO_TEMPLATE", "1")
	const output = "completion fixture"
	var finalCounts int
	mock := &mockRunner{
		CompletionResponse: llm.CompletionResponse{Content: output, Done: true, DoneReason: llm.DoneReasonStop},
		TokenizeFn: func(_ context.Context, prompt string) ([]int, error) {
			if strings.HasSuffix(prompt, output) {
				finalCounts++
			}
			return []int{31, 32}, nil
		},
	}
	s := newServerWithMockRunner(t, mock)
	createMinimalGGUFModel(t, s, "fixture", nil, "{{ .Prompt }}", nil)
	router := gin.New()
	router.POST("/v1/completions", middleware.CompletionsMiddleware(), s.GenerateHandler)
	router.POST("/api/generate", s.GenerateHandler)
	for _, stream := range []bool{false, true} {
		for _, path := range []string{"/v1/completions", "/api/generate"} {
			body := fmt.Sprintf(`{"model":"fixture","prompt":"hello","stream":%t}`, stream)
			prior := finalCounts
			w := NewRecorder()
			router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
			want := 0
			if path == "/api/generate" {
				want = 1
			}
			if finalCounts-prior != want || w.Code != 200 || !strings.Contains(w.Body.String(), output) || strings.Contains(w.Body.String(), `"context":[31,32]`) != (want == 1) {
				t.Fatalf("%s stream=%t: final tokenizations=%d, result=%s", path, stream, finalCounts-prior, w.Body.String())
			}
		}
	}
}

func BenchmarkCompatResponseWire(b *testing.B) {
	gin.SetMode(gin.TestMode)
	for _, mode := range []string{"chat-256-chunks", "embed-float", "embed-base64"} {
		b.Run(mode, func(b *testing.B) {
			router := gin.New()
			var request string
			if mode == "chat-256-chunks" {
				request = `{"model":"fixture","messages":[{"role":"user","content":"test"}],"stream":true}`
				router.POST("/", middleware.ChatMiddleware(), func(c *gin.Context) {
					frames := make(chan any, 257)
					chunk := api.ChatResponse{Model: "fixture", CreatedAt: time.Unix(100, 0), Message: api.Message{Role: "assistant", Content: "aÃ§Ã£o"}}
					for range 256 {
						frames <- chunk
					}
					frames <- api.ChatResponse{Model: "fixture", Done: true, DoneReason: "stop"}
					close(frames)
					streamResponse(c, frames)
				})
			} else {
				format := strings.TrimPrefix(mode, "embed-")
				request = fmt.Sprintf(`{"model":"fixture","input":"text","encoding_format":%q}`, format)
				result := api.EmbedResponse{Model: "fixture", PromptEvalCount: 500, Embeddings: make([][]float32, 16)}
				for i := range result.Embeddings {
					result.Embeddings[i] = make([]float32, 2048)
					for k := range result.Embeddings[i] {
						result.Embeddings[i][k] = float32(math.Sin(float64(k+i))) * 0.01
					}
				}
				router.POST("/", middleware.EmbeddingsMiddleware(), func(c *gin.Context) { writeModelResponse(c, result) })
			}
			b.ReportAllocs()
			for b.Loop() {
				w := NewRecorder()
				r := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(request))
				router.ServeHTTP(w, r)
				if w.Code != 200 || w.Body.Len() == 0 {
					b.Fatalf("empty response (%d)", w.Code)
				}
			}
		})
	}
}

func TestCompatPartialStreamPreservesError(t *testing.T) {
	for _, chat := range []bool{true, false} {
		router := gin.New()
		adapter := middleware.CompletionsMiddleware()
		request := `{"model":"fixture","prompt":"test","stream":true}`
		var partial any = api.GenerateResponse{Model: "fixture", Response: "partial"}
		if chat {
			adapter = middleware.ChatMiddleware()
			request = `{"model":"fixture","messages":[{"role":"user","content":"test"}],"stream":true}`
			partial = api.ChatResponse{Model: "fixture", Message: api.Message{Role: "assistant", Content: "partial"}}
		}
		router.POST("/", adapter, func(c *gin.Context) {
			frames := make(chan any, 2)
			frames <- partial
			frames <- gin.H{"error": "fixture ran out of memory", "status": 507, "code": "out_of_memory", "retryable": false}
			close(frames)
			streamResponse(c, frames)
		})
		w := NewRecorder()
		router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(request)))
		data := strings.Split(strings.TrimSpace(w.Body.String()), "\n\n")
		var failure struct {
			Error struct{ Message, Code string }
		}
		if len(data) != 2 || !strings.Contains(data[0], "partial") || json.Unmarshal([]byte(strings.TrimPrefix(data[1], "data: ")), &failure) != nil || failure.Error.Message != "fixture ran out of memory" || failure.Error.Code != "out_of_memory" {
			t.Errorf("chat=%t invalid stream: %s", chat, w.Body)
		}
	}
}

func TestCompatEmbeddingResponseMatchesByteAdapter(t *testing.T) {
	result := api.EmbedResponse{Model: "fixture", PromptEvalCount: 7, Embeddings: [][]float32{
		{math.SmallestNonzeroFloat32, -0.25}, {math.Float32frombits(1 << 31), 0.5},
	}}
	for _, format := range []string{"float", "base64"} {
		var output [2]string
		for i := range output {
			router := gin.New()
			router.POST("/", middleware.EmbeddingsMiddleware(), func(c *gin.Context) {
				if i == 0 {
					c.JSON(http.StatusOK, result)
				} else {
					writeModelResponse(c, result)
				}
			})
			body := fmt.Sprintf(`{"model":"fixture","input":["first","second"],"encoding_format":%q}`, format)
			w := NewRecorder()
			router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body)))
			if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"prompt_tokens":7`) {
				t.Fatalf("invalid %s response: %s", format, w.Body.String())
			}
			output[i] = w.Body.String()
		}
		if output[0] != output[1] {
			t.Fatalf("%s: byte and typed responses differ", format)
		}
	}
}

// The wrapper exposes only Gin's writer methods and forces the byte adapter.
type compatByteOnlyWriter struct{ gin.ResponseWriter }

func TestNativeMessageProtocolsMatchByteAdapter(t *testing.T) {
	requestID := regexp.MustCompile(`"(resp_|msg_|rs_|fc_)[a-zA-Z0-9_]+"`)
	timestamp := regexp.MustCompile(`("(?:created_at|completed_at)":)[0-9]+`)
	for _, p := range []struct {
		name, request, keepalive, done, failed string
		middleware                             gin.HandlerFunc
	}{
		{"Responses", `{"model":"test","input":"olá","stream":%t}`, ": generation_progress", "response.completed", "response.failed", middleware.ResponsesMiddleware()},
		{"Messages", `{"model":"test","max_tokens":16,"messages":[{"role":"user","content":"olá"}],"stream":%t}`, "event: ping", "event: message_stop", "event: error", middleware.AnthropicMessagesMiddleware()},
	} {
		for _, mode := range []string{"stream", "failure", "complete"} {
			t.Run(p.name+"/"+mode, func(t *testing.T) {
				args := api.NewToolCallFunctionArguments()
				args.Set("id", int64(9007199254740993))
				end := api.ChatResponse{
					Model: "test", CreatedAt: time.Unix(100, 0), Done: true, DoneReason: "stop",
					Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{
						{ID: "call_fixed", Function: api.ToolCallFunction{Name: "pesquisar", Arguments: args}},
					}},
					Metrics: api.Metrics{PromptEvalCount: 7, EvalCount: 3},
				}
				var outputs [2]string
				for i := range outputs {
					router := gin.New()
					router.POST("/", p.middleware, func(c *gin.Context) {
						if i == 1 {
							c.Writer = compatByteOnlyWriter{c.Writer}
						}
						if mode == "complete" {
							writeModelResponse(c, end)
							return
						}
						frames := make(chan any, 4)
						frames <- api.ChatResponse{Model: "test", GenerationProgress: &api.GenerationProgress{Phase: "tool_call"}}
						frames <- api.ChatResponse{Model: "test", Message: api.Message{Role: "assistant", Thinking: "pensando"}}
						frames <- api.ChatResponse{Model: "test", Message: api.Message{Role: "assistant", Content: "ação & café"}}
						if mode == "failure" {
							frames <- gin.H{"error": "falha controlada", "status": 503}
						} else {
							frames <- end
						}
						close(frames)
						streamResponse(c, frames)
					})
					w := NewRecorder()
					router.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(fmt.Sprintf(p.request, mode != "complete"))))
					body := w.Body.String()
					if w.Code != 200 || (mode != "complete" && (!strings.Contains(body, p.keepalive) || !strings.Contains(body, "pensando") || !strings.Contains(body, "caf"))) {
						t.Fatalf("invalid response: %d %s", w.Code, body)
					}
					if mode == "failure" {
						if !strings.Contains(body, p.failed) || !strings.Contains(body, "falha controlada") {
							t.Fatal(body)
						}
					} else if !strings.Contains(body, "pesquisar") || !strings.Contains(body, "9007199254740993") || (mode == "stream" && !strings.Contains(body, p.done)) {
						t.Fatal(body)
					}
					// Request IDs and wall-clock timestamps are generated independently.
					body = requestID.ReplaceAllString(body, `"${1}id"`)
					outputs[i] = timestamp.ReplaceAllString(body, `${1}0`)
				}
				if outputs[0] != outputs[1] {
					t.Fatalf("native and byte responses differ:\n%s\n%s", outputs[0], outputs[1])
				}
			})
		}
	}
}
