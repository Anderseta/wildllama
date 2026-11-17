package llm

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
)

func TestNativeToolIntegerRoundTrip(t *testing.T) {
	const arguments = `{"seed":5232021495785965240,"nested":[-9223372036854775808,{"id":18446744073709551615}],"ordinary":42,"text":"5232021495785965240"}`
	var acc llamaServerToolCallAccumulator
	acc.append("call", "fixture", arguments[:12])
	acc.append("", "", arguments[12:])
	calls, err := accumulatedToolCalls(map[int]*llamaServerToolCallAccumulator{0: &acc}, false)
	if err != nil {
		t.Fatal(err)
	}
	history, err := llamaServerChatToolCalls(calls)
	if err != nil {
		t.Fatal(err)
	}
	if got := history[0].Function.Arguments; got != arguments {
		t.Fatalf("tool history changed numeric arguments: %s", got)
	}
}

func BenchmarkNativeChatHistoryEncoding(b *testing.B) {
	runner := &llamaServerRunner{}
	messages := make([]api.Message, 0, 1024)
	for range 512 {
		messages = append(messages,
			api.Message{Role: "user", Content: "Check the following procedure."},
			api.Message{Role: "assistant", Content: "The check passed.", Thinking: "I verified the inputs."})
	}
	req := ChatRequest{Messages: messages}
	b.ReportAllocs()
	for b.Loop() {
		body, err := runner.llamaServerChatRequest(req, true)
		if err != nil {
			b.Fatal(err)
		}
		if _, err = encodeLlamaServerChatRequest(body); err != nil {
			b.Fatal(err)
		}
	}
}

func TestNativeChatPreservesAssistantReasoningOnly(t *testing.T) {
	for _, role := range []string{"assistant", "user", "system", "tool"} {
		t.Run(role, func(t *testing.T) {
			message := Message{Role: role, Content: "visible answer", Thinking: "  reasoning before a tool\n"}
			converted, err := llamaServerChatMessage(message)
			if err != nil {
				t.Fatal(err)
			}
			body, err := json.Marshal(converted)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]any
			if err := json.Unmarshal(body, &fields); err != nil {
				t.Fatal(err)
			}
			reasoning, exists := fields["reasoning_content"]
			if role == "assistant" {
				if !exists || reasoning != message.Thinking {
					t.Fatalf("assistant reasoning was not preserved byte-for-byte: %#v", reasoning)
				}
			} else if exists {
				t.Fatalf("%s must not acquire assistant reasoning", role)
			}
			if fields["content"] != message.Content {
				t.Fatal("reasoning conversion changed visible content")
			}
		})
	}
	converted, err := llamaServerChatMessage(Message{Role: "assistant"})
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(converted)
	if err != nil || string(body) != `{"content":"","role":"assistant"}` {
		t.Fatalf("unexpected empty assistant message %s: %v", body, err)
	}
}

func TestNativeChatReasoningSurvivesToolHistorySerialization(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("path", "example.txt")
	opts := api.DefaultOptions()
	runner := &llamaServerRunner{}
	body, err := runner.llamaServerChatRequest(ChatRequest{
		Options: &opts,
		Messages: []api.Message{
			{Role: "user", Content: "inspect the file"},
			{Role: "assistant", Thinking: "First inspect the relevant file.", ToolCalls: []api.ToolCall{{
				ID: "call-1", Function: api.ToolCallFunction{Name: "read", Arguments: args},
			}}},
			{Role: "tool", ToolCallID: "call-1", Content: "example contents"},
		},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Messages []struct {
			Reasoning string            `json:"reasoning_content"`
			Calls     []json.RawMessage `json:"tool_calls"`
			CallID    string            `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Messages[1].Reasoning != "First inspect the relevant file." || len(decoded.Messages[1].Calls) != 1 || decoded.Messages[2].CallID != "call-1" {
		t.Fatal("native tool follow-up lost reasoning, its call, or the matching result")
	}
}

func TestNativeChatTemplateTokenCountCompatibility(t *testing.T) {
	for _, test := range []struct {
		name       string
		response   string
		wantCount  int
		wantTokens int
		wantError  bool
	}{
		{"combined count", `{"prompt":"<s>rendered prompt","token_count":17}`, 17, 0, false},
		{"valid zero", `{"prompt":"","token_count":0}`, 0, 0, false},
		{"older payload", `{"prompt":"<s>rendered prompt"}`, 3, 1, false},
		{"invalid count", `{"prompt":"bad","token_count":-1}`, 0, 0, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			applyCalls, tokenizeCalls := 0, 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch r.URL.Path {
				case "/apply-template":
					applyCalls++
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					if body["count_tokens"] != true {
						t.Error("missing additive count_tokens request")
					}
					_, _ = w.Write([]byte(test.response))
				case "/tokenize":
					tokenizeCalls++
					var body struct {
						Content    string `json:"content"`
						AddSpecial bool   `json:"add_special"`
					}
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					if !body.AddSpecial || body.Content != "<s>rendered prompt" {
						t.Error("fallback must use the unchanged full prompt and generation's BOS policy")
					}
					_, _ = w.Write([]byte(`{"tokens":[1,2,3]}`))
				default:
					t.Errorf("unexpected endpoint %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			runner := &llamaServerRunner{port: server.Listener.Addr().(*net.TCPAddr).Port}
			opts := api.DefaultOptions()
			_, count, err := runner.ApplyChatTemplateWithTokenCount(context.Background(), ChatRequest{
				Options: &opts, Messages: []api.Message{{Role: "user", Content: "hello"}},
			})
			if (err != nil) != test.wantError || count != test.wantCount {
				t.Fatalf("count=%d, err=%v; want count=%d, error=%v", count, err, test.wantCount, test.wantError)
			}
			if applyCalls != 1 || tokenizeCalls != test.wantTokens {
				t.Fatalf("HTTP calls apply=%d tokenize=%d; expected 1 and %d", applyCalls, tokenizeCalls, test.wantTokens)
			}
		})
	}
}

func TestNativeChatTemplateCountDoesNotRetryInvalidTemplate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/apply-template" {
			t.Errorf("invalid template must not be retried through %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid tool history"}}`))
	}))
	defer server.Close()
	runner := &llamaServerRunner{port: server.Listener.Addr().(*net.TCPAddr).Port}
	opts := api.DefaultOptions()
	_, _, err := runner.ApplyChatTemplateWithTokenCount(context.Background(), ChatRequest{Options: &opts})
	if err == nil || !strings.Contains(err.Error(), "invalid tool history") {
		t.Fatalf("template error was hidden: %v", err)
	}
}

func TestNativeChatPreparedBodyMatchesGeneration(t *testing.T) {
	for _, prepared := range []bool{false, true} {
		name := "legacy runner"
		if prepared {
			name = "prepared chat"
		}
		t.Run(name, func(t *testing.T) {
			const key = "ONE_PRIVATE_PREPARATION_KEY"
			var mutex sync.Mutex
			var preparedBody string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					_, _ = io.WriteString(w, `{"status":"ok"}`)
					return
				}
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				mutex.Lock()
				defer mutex.Unlock()
				switch r.URL.Path {
				case "/apply-template":
					preparedBody = string(body)
					if r.Header.Get(preparedChatHeader) != key || r.Header.Get("X-Wildllama-Chat-Token-Limit") != "64" {
						t.Error("missing runner preparation identity or budget")
					}
					if prepared {
						_, _ = io.WriteString(w, `{"token_count":17,"prepared":true}`)
					} else {
						_, _ = io.WriteString(w, `{"prompt":"legacy","token_count":17}`)
					}
				case "/v1/chat/completions":
					if prepared && (r.Header.Get(preparedChatHeader) != key || string(body) != preparedBody) {
						t.Error("generation differs from prepared streaming body")
					}
					if !prepared && r.Header.Get(preparedChatHeader) != "" {
						t.Error("generation sent a preparation key rejected by the runner")
					}
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}],\"timings\":{\"prompt_n\":17,\"predicted_n\":1}}\n\ndata: [DONE]\n\n")
				default:
					t.Errorf("unexpected endpoint %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			runner := &llamaServerRunner{
				port:    srv.Listener.Addr().(*net.TCPAddr).Port,
				cmd:     fakeRunningCmd(),
				sem:     semaphore.NewWeighted(1),
				options: api.Options{Runner: api.Runner{NumCtx: 128}},
			}
			opts := api.DefaultOptions()
			opts.NumPredict = 1000
			req := ChatRequest{
				Options:  &opts,
				Messages: []api.Message{{Role: "system", Content: "<instructions>"}, {Role: "user", Content: "Parâmetros & Unicode."}},
			}
			count, accepted, err := runner.PrepareChatSnapshot(t.Context(), req, key, 64)
			if err != nil || count != 17 || (accepted != nil) != prepared {
				t.Fatalf("count=%d accepted=%t err=%v", count, accepted != nil, err)
			}
			if accepted != nil {
				// Sending the accepted snapshot is independent of any later edits.
				req.Messages[1].Content = "different request"
				err = accepted(t.Context(), func(ChatResponse) {})
			} else {
				err = runner.Chat(t.Context(), req, func(ChatResponse) {})
			}
			if err != nil {
				t.Fatal(err)
			}
			// Independently sized clients can still submit their own request.
			if prepared {
				req.PreparedChat = key
				_, ok, err := runner.PrepareChatTemplateWithTokenCount(t.Context(), req, 64)
				if err != nil || !ok {
					t.Fatalf("key-based API: accepted=%t error=%v", ok, err)
				}
				if err := runner.Chat(t.Context(), req, func(ChatResponse) {}); err != nil {
					t.Fatal(err)
				}
			}
			if opts.NumPredict != 1000 {
				t.Fatal("preparation or generation changed the caller's options")
			}
		})
	}
}

func TestNativeResponseSchemaExactLiterals(t *testing.T) {
	const schema = `{"type":"object","properties":{"z":{"const":5232021495785965240},"a":{"enum":[18446744073709551615]}}}`
	format, err := llamaServerChatResponseFormat(json.RawMessage(schema))
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(format)
	if err != nil || !strings.Contains(string(encoded), schema) {
		t.Fatalf("schema changed on its way to the runner: %s, %v", encoded, err)
	}
}

func BenchmarkPreparedNativeChatRoundTrip(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		switch r.URL.Path {
		case "/health":
			io.WriteString(w, `{"status":"ok"}`)
		case "/apply-template":
			io.WriteString(w, `{"prepared":true,"token_count":12000}`)
		default:
			io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
		}
	}))
	defer server.Close()
	runner := &llamaServerRunner{port: server.Listener.Addr().(*net.TCPAddr).Port, cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1), options: api.Options{Runner: api.Runner{NumCtx: 40960}}}
	msgs := make([]api.Message, 1024)
	for i := range msgs {
		msgs[i] = api.Message{Role: "user", Content: strings.Repeat("example <source> & UTF-8 ação ", 4)}
	}
	req := ChatRequest{Messages: msgs}
	b.ReportAllocs()
	for b.Loop() {
		_, submit, err := runner.PrepareChatSnapshot(b.Context(), req, "test_preparation_key", 40960)
		if err != nil || submit == nil {
			b.Fatalf("preparation: %t, %v", submit != nil, err)
		}
		if err := submit(b.Context(), func(ChatResponse) {}); err != nil {
			b.Fatal(err)
		}
	}
}
