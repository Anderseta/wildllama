package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
)

func TestChatBufferedGenerationProgress(t *testing.T) {
	for _, mode := range []string{"native", "generic", "builtin"} {
		for _, stream := range []bool{true, false} {
			t.Run(mode+"/"+map[bool]string{true: "stream", false: "nonstream"}[stream], func(t *testing.T) {
				t.Setenv("OLLAMA_MODELS", t.TempDir())
				t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
				t.Setenv("OLLAMA_GO_TEMPLATE", "")
				gin.SetMode(gin.TestMode)
				payload := "begin\n" + strings.Repeat("texto com ação e código\n", 1000) + "end"
				args := api.NewToolCallFunctionArguments()
				args.Set("content", payload)
				call := api.ToolCall{ID: "call_write", Function: api.ToolCallFunction{Name: "write_file", Arguments: args}}
				fixture := &mockRunner{
					ChatFn: func(_ context.Context, _ llm.ChatRequest, fn func(llm.ChatResponse)) error {
						fn(llm.ChatResponse{GenerationProgress: &api.GenerationProgress{Phase: "tool_call"}})
						fn(llm.ChatResponse{Message: api.Message{ToolCalls: []api.ToolCall{call}}, Done: true, EvalCount: 7})
						return nil
					},
					CompletionFn: func(_ context.Context, _ llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
						// An empty callback cannot fabricate model activity.
						fn(llm.CompletionResponse{})
						var start, body, end string
						if mode == "builtin" {
							start = "<tool_call><function=write_file><parameter=content>"
							body = payload
							end = "</parameter></function></tool_call>"
						} else {
							start = `<tool_call>{"name":"write_file","arguments":`
							encoded, err := json.Marshal(map[string]string{"content": payload})
							if err != nil {
								return err
							}
							body, end = string(encoded), `}</tool_call>`
						}
						fn(llm.CompletionResponse{Content: start})
						for offset := 0; offset < len(body); offset += 100 {
							fn(llm.CompletionResponse{Content: body[offset:min(offset+100, len(body))]})
						}
						fn(llm.CompletionResponse{Content: end})
						fn(llm.CompletionResponse{Done: true, EvalCount: 7})
						return nil
					},
				}
				s := newServerWithMockRunner(t, fixture)
				kv := gguftest.KV{"tokenizer.chat_template": "{{ tools }}{{ messages[0]['content'] }}"}
				tmpl := ""
				if mode != "native" {
					t.Setenv("OLLAMA_GO_TEMPLATE", "1")
					tmpl = `{{range .Messages}}{{.Content}}{{if .ToolCalls}}<tool_call>{{range .ToolCalls}}{"name":"{{.Function.Name}}","arguments":{{.Function.Arguments}}}{{end}}</tool_call>{{end}}{{end}}{{if .Tools}}tools{{end}}`
				}
				createMinimalGGUFModel(t, s, "progress-base", kv, tmpl, nil)
				name := "progress-base"
				if mode == "builtin" {
					name = "progress-builtin"
					nonstream := false
					w := createRequest(t, s.CreateHandler, api.CreateRequest{Model: name, From: "progress-base", Parser: "qwen3.5", Stream: &nonstream})
					if w.Code != http.StatusOK {
						t.Fatalf("create parser model: %d %s", w.Code, w.Body.String())
					}
				}
				w := createRequest(t, s.ChatHandler, api.ChatRequest{
					Model: name, Messages: []api.Message{{Role: "user", Content: "write a file"}}, Stream: &stream,
					Think: &api.ThinkValue{Value: false},
					Tools: []api.Tool{{Type: "function", Function: api.ToolFunction{Name: "write_file", Parameters: api.ToolFunctionParameters{Type: "object"}}}},
				})
				if w.Code != http.StatusOK {
					t.Fatalf("chat: %d %s", w.Code, w.Body.String())
				}
				decoder := json.NewDecoder(w.Body)
				var progressCount, doneCount int
				var calls []api.ToolCall
				for {
					var frame api.ChatResponse
					if err := decoder.Decode(&frame); err == io.EOF {
						break
					} else if err != nil {
						t.Fatal(err)
					}
					if frame.GenerationProgress != nil {
						progressCount++
						if frame.Done || len(frame.Message.ToolCalls) != 0 || frame.EvalCount != 0 || frame.GenerationProgress.Phase == "" {
							t.Fatalf("progress exposed an incomplete call or token count: %+v", frame)
						}
					}
					if frame.Message.Content != "" || frame.Message.Thinking != "" {
						t.Fatal("tool arguments leaked as conversational output")
					}
					calls = append(calls, frame.Message.ToolCalls...)
					if frame.Done {
						doneCount++
						if frame.EvalCount != 7 || frame.GenerationProgress != nil {
							t.Fatalf("terminal metrics changed: %+v", frame)
						}
					}
				}
				wantProgress := 0
				if stream {
					wantProgress = 1
				}
				if progressCount != wantProgress || doneCount != 1 || len(calls) != 1 {
					t.Fatalf("progress=%d done=%d tools=%d", progressCount, doneCount, len(calls))
				}
				got, ok := calls[0].Function.Arguments.Get("content")
				if calls[0].Function.Name != "write_file" || !ok || got != payload {
					t.Fatal("completed tool payload was changed")
				}
			})
		}
	}
}
