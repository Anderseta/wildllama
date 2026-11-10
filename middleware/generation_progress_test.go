package middleware

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/anthropic"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/openai"
)

type generationProgressProtocol struct {
	name, endpoint, request, keepalive, firstEvent, lastEvent string
	middleware                                                gin.HandlerFunc
	writer                                                    func(gin.ResponseWriter, bool) io.Writer
}

func generationProgressProtocols() []generationProgressProtocol {
	return []generationProgressProtocol{
		{
			name: "Chat", endpoint: "/v1/chat/completions",
			request:   `{"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true}`,
			keepalive: ": generation_progress\n\n", middleware: ChatMiddleware(),
			writer: func(w gin.ResponseWriter, stream bool) io.Writer {
				return &ChatWriter{BaseWriter: BaseWriter{ResponseWriter: w}, stream: stream, id: "chat_fixture"}
			},
		},
		{
			name: "Responses", endpoint: "/v1/responses",
			request:   `{"model":"test","input":"hello","stream":true}`,
			keepalive: ": generation_progress\n\n", firstEvent: "response.created", lastEvent: "response.completed",
			middleware: ResponsesMiddleware(),
			writer: func(w gin.ResponseWriter, stream bool) io.Writer {
				return &ResponsesWriter{BaseWriter: BaseWriter{ResponseWriter: w}, stream: stream, model: "test",
					responseID: "resp_fixture", itemID: "item_fixture", request: openai.ResponsesRequest{Model: "test"},
					converter: openai.NewResponsesStreamConverter("resp_fixture", "item_fixture", "test", openai.ResponsesRequest{Model: "test"})}
			},
		},
		{
			name: "Anthropic", endpoint: "/v1/messages",
			request:   `{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"hello"}],"stream":true}`,
			keepalive: "event: ping\ndata: {\"type\":\"ping\"}\n\n", firstEvent: "message_start", lastEvent: "message_stop",
			middleware: AnthropicMessagesMiddleware(),
			writer: func(w gin.ResponseWriter, stream bool) io.Writer {
				return &AnthropicWriter{BaseWriter: BaseWriter{ResponseWriter: w}, stream: stream, id: "msg_fixture",
					converter: anthropic.NewStreamConverter("msg_fixture", "test", 0)}
			},
		},
	}
}

func generationProgressFrame(phase string) api.ChatResponse {
	return api.ChatResponse{Model: "test", Message: api.Message{Role: "assistant"}, GenerationProgress: &api.GenerationProgress{Phase: phase}}
}

func writeGenerationProgressFrame(t *testing.T, writer io.Writer, frame api.ChatResponse) {
	t.Helper()
	data, err := json.Marshal(frame)
	if err != nil {
		t.Fatal(err)
	}
	if n, err := writer.Write(data); err != nil || n != len(data) {
		t.Fatalf("write = %d, %v; want %d, nil", n, err, len(data))
	}
}

func TestBufferedGenerationProgressProtocolStreams(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, protocol := range generationProgressProtocols() {
		t.Run(protocol.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			router := gin.New()
			router.POST(protocol.endpoint, protocol.middleware, func(c *gin.Context) {
				// Repeated activity before the first output must not open a message,
				// create tool/content blocks or consume Responses sequence numbers.
				for _, phase := range []string{"tool_call", "buffered_output"} {
					writeGenerationProgressFrame(t, c.Writer, generationProgressFrame(phase))
				}
				if got, want := recorder.Body.String(), strings.Repeat(protocol.keepalive, 2); got != want {
					t.Fatalf("progress advanced protocol lifecycle: %q, want %q", got, want)
				}
				if !recorder.Flushed {
					t.Fatal("progress was not flushed to the client")
				}

				thinking := generationProgressFrame("buffered_output")
				thinking.Message.Thinking = "Inspecting the file."
				thinking.CreatedAt = time.Unix(123, 0)
				writeGenerationProgressFrame(t, c.Writer, thinking)
				before := recorder.Body.Len()
				for _, phase := range []string{"tool_call", "buffered_output"} {
					writeGenerationProgressFrame(t, c.Writer, generationProgressFrame(phase))
				}
				if got, want := recorder.Body.String()[before:], strings.Repeat(protocol.keepalive, 2); got != want {
					t.Fatalf("progress changed an open block: %q, want %q", got, want)
				}

				final := generationProgressFrame("tool_call")
				final.Done, final.DoneReason = true, "stop"
				final.CreatedAt = thinking.CreatedAt
				final.Message.ToolCalls = []api.ToolCall{{ID: "call_fixture", Function: api.ToolCallFunction{
					Name: "write_file", Arguments: testArgs(map[string]any{"path": "example.txt", "content": "file contents"}),
				}}}
				final.Metrics = api.Metrics{PromptEvalCount: 12, EvalCount: 7}
				writeGenerationProgressFrame(t, c.Writer, final)
			})
			request := httptest.NewRequest(http.MethodPost, protocol.endpoint, strings.NewReader(protocol.request))
			request.Header.Set("Content-Type", "application/json")
			router.ServeHTTP(recorder, request)
			body := recorder.Body.String()
			if recorder.Code != http.StatusOK || recorder.Header().Get("Content-Type") != "text/event-stream" {
				t.Fatalf("status=%d content-type=%q: %s", recorder.Code, recorder.Header().Get("Content-Type"), body)
			}
			if protocol.name != "Chat" && (strings.Count(body, "event: "+protocol.firstEvent+"\n") != 1 || strings.Count(body, "event: "+protocol.lastEvent+"\n") != 1) {
				t.Fatalf("message lifecycle duplicated or missing: %s", body)
			}
			if protocol.name == "Chat" && (strings.Count(body, `"role":"assistant"`) != 1 || strings.Count(body, `"finish_reason":"tool_calls"`) != 1 || strings.Count(body, "data: [DONE]") != 1) {
				t.Fatalf("chat role or tool lifecycle changed: %s", body)
			}
			if !strings.Contains(body, "file contents") || !strings.Contains(body, "Inspecting the file.") {
				t.Fatalf("complete tool/thinking output changed: %s", body)
			}
			sequence := float64(0)
			toolStarts := 0
			for _, line := range strings.Split(body, "\n") {
				if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
					continue
				}
				var event map[string]any
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
					t.Fatal(err)
				}
				if protocol.name == "Responses" {
					if event["sequence_number"] != sequence {
						t.Fatalf("response sequence=%v, want %v", event["sequence_number"], sequence)
					}
					sequence++
				}
				switch protocol.name {
				case "Responses":
					if event["type"] == "response.output_item.added" {
						if item, ok := event["item"].(map[string]any); ok && item["type"] == "function_call" && item["name"] == "write_file" {
							toolStarts++
						}
					}
				case "Anthropic":
					if event["type"] == "content_block_start" {
						if block, ok := event["content_block"].(map[string]any); ok && block["type"] == "tool_use" && block["name"] == "write_file" {
							toolStarts++
						}
					}
				case "Chat":
					if event["created"] != float64(123) {
						t.Fatalf("progress pinned timestamp before actual output: %v", event["created"])
					}
					for _, choice := range event["choices"].([]any) {
						delta := choice.(map[string]any)["delta"].(map[string]any)
						if calls, ok := delta["tool_calls"].([]any); ok {
							toolStarts += len(calls)
						}
					}
				}
			}
			if toolStarts != 1 {
				t.Fatalf("tool starts = %d, want 1: %s", toolStarts, body)
			}
		})
	}
}

func TestBufferedGenerationProgressPreservesNonstreamAndErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, protocol := range generationProgressProtocols() {
		t.Run(protocol.name+" nonstream", func(t *testing.T) {
			recorder := httptest.NewRecorder()
			ctx, _ := gin.CreateTestContext(recorder)
			writer := protocol.writer(ctx.Writer, false)
			final := generationProgressFrame("tool_call")
			final.Done, final.DoneReason, final.Message.Content = true, "stop", "FINAL_TEXT"
			writeGenerationProgressFrame(t, writer, final)
			if !json.Valid(recorder.Body.Bytes()) || !strings.Contains(recorder.Body.String(), "FINAL_TEXT") || strings.Contains(recorder.Body.String(), "generation_progress") || strings.Contains(recorder.Body.String(), "event:") {
				t.Fatalf("nonstream response changed: %s", recorder.Body.String())
			}
		})
		t.Run(protocol.name+" error takes precedence", func(t *testing.T) {
			recorder := httptest.NewRecorder()
			ctx, _ := gin.CreateTestContext(recorder)
			writer := protocol.writer(ctx.Writer, true)
			if protocol.name == "Chat" {
				ctx.Writer.WriteHeader(http.StatusInternalServerError)
			}
			data := []byte(`{"error":"RUNNER_FAILED","message":{"role":"assistant"},"generation_progress":{"phase":"tool_call"}}`)
			if _, err := writer.Write(data); err != nil {
				t.Fatal(err)
			}
			if body := recorder.Body.String(); !strings.Contains(body, "RUNNER_FAILED") || strings.Contains(body, protocol.keepalive) {
				t.Fatalf("progress concealed an error: %s", body)
			}
		})
		t.Run(protocol.name+" cancelled write", func(t *testing.T) {
			recorder := httptest.NewRecorder()
			ctx, _ := gin.CreateTestContext(recorder)
			closed := &generationProgressClosedWriter{ResponseWriter: ctx.Writer}
			writer := protocol.writer(closed, true)
			data, err := json.Marshal(generationProgressFrame("tool_call"))
			if err != nil {
				t.Fatal(err)
			}
			if n, err := writer.Write(data); n != 0 || !errors.Is(err, context.Canceled) {
				t.Fatalf("cancelled write = %d, %v; want 0, context.Canceled", n, err)
			}
			if recorder.Body.Len() != 0 || recorder.Flushed {
				t.Fatal("cancelled progress wrote or flushed output")
			}
		})
	}
}

type generationProgressClosedWriter struct{ gin.ResponseWriter }

func (*generationProgressClosedWriter) Write([]byte) (int, error) { return 0, context.Canceled }
func (*generationProgressClosedWriter) WriteString(string) (int, error) {
	return 0, context.Canceled
}
