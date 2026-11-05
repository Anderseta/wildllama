package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/middleware"
	"github.com/ollama/ollama/types/model"
)

func TestNativeThinkingDiscoveryAndRequests(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_GO_TEMPLATE", "0")
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	mock := mockRunner{ChatFn: func(_ context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
		fn(llm.ChatResponse{Message: api.Message{Role: "assistant", Content: "answer"}, Done: true, DoneReason: llm.DoneReasonStop})
		return nil
	}}
	s := newServerWithMockRunner(t, &mock)
	for _, tc := range []struct {
		name, fixture string
		want          *model.Thinking
	}{
		{"native-effort", "../model/renderers/testdata/qwen38_chat_template.jinja", &model.Thinking{Values: []any{false, "xhigh", "medium", "low"}, Default: "xhigh"}},
		{"native-toggle", "testdata/qwen35_chat_template.jinja", &model.Thinking{Values: []any{false, true}, Default: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source, err := os.ReadFile(tc.fixture)
			if err != nil {
				t.Fatal(err)
			}
			createMinimalGGUFModel(t, s, tc.name, gguftest.KV{"tokenizer.chat_template": string(source)}, "", nil)
			show, err := GetModelInfo(api.ShowRequest{Model: tc.name})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(show.Thinking, tc.want) {
				t.Fatalf("thinking metadata = %#v, want %#v", show.Thinking, tc.want)
			}
			for _, requested := range []any{nil, false, true, "low", "medium", "xhigh", "unsupported"} {
				for _, endpoint := range []string{"chat", "generate"} {
					t.Run(fmt.Sprintf("%s/%v", endpoint, requested), func(t *testing.T) {
						var think *api.ThinkValue
						if requested != nil {
							think = &api.ThinkValue{Value: requested}
						}
						var w *httptest.ResponseRecorder
						if endpoint == "chat" {
							w = createRequest(t, s.ChatHandler, api.ChatRequest{Model: tc.name, Messages: []api.Message{{Role: "user", Content: "hello"}}, Think: think, Stream: &stream})
						} else {
							w = createRequest(t, s.GenerateHandler, api.GenerateRequest{Model: tc.name, Prompt: "hello", Think: think, Stream: &stream})
						}
						if w.Code != http.StatusOK {
							t.Fatalf("status=%d: %s", w.Code, w.Body.String())
						}
						want := requested
						if want == nil || !tc.want.Supports(want) {
							if _, boolean := want.(bool); !boolean {
								want = tc.want.Default
							}
						}
						if mock.ChatRequest.Think == nil || mock.ChatRequest.Think.Value != want {
							t.Fatalf("native request think=%#v, want %#v", mock.ChatRequest.Think, want)
						}
					})
				}
			}
		})
	}
}

func TestNativeThinkingCompatibilityKeepsExactEffort(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_GO_TEMPLATE", "0")
	source, err := os.ReadFile("../model/renderers/testdata/qwen38_chat_template.jinja")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{}
	createMinimalGGUFModel(t, s, "native-effort", gguftest.KV{"tokenizer.chat_template": string(source)}, "", nil)
	for _, protocol := range []struct {
		name, fields string
		handler      gin.HandlerFunc
	}{
		{"chat", `"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"xhigh"`, middleware.ChatMiddleware(lookupThinking)},
		{"responses", `"input":"hi","reasoning":{"effort":"xhigh"}`, middleware.ResponsesMiddleware(lookupThinking)},
		{"anthropic", `"messages":[{"role":"user","content":"hi"}],"max_tokens":64,"output_config":{"effort":"xhigh"}`, middleware.AnthropicMessagesMiddleware(lookupThinking)},
	} {
		t.Run(protocol.name, func(t *testing.T) {
			var req api.ChatRequest
			router := gin.New()
			router.POST("/", protocol.handler, func(c *gin.Context) {
				if err := json.NewDecoder(c.Request.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				c.Status(http.StatusOK)
			})
			request := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"model":"native-effort",`+protocol.fields+`}`))
			request.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			router.ServeHTTP(w, request)
			if w.Code != http.StatusOK || req.Think == nil || req.Think.Value != "xhigh" {
				t.Fatalf("status=%d think=%#v: %s", w.Code, req.Think, w.Body.String())
			}
		})
	}
}
