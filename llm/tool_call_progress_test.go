package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func toolProgressTestRunner(t *testing.T, stream func(http.ResponseWriter, *http.Request)) *llamaServerRunner {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			fmt.Fprint(w, `{"status":"ok"}`)
			return
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected native endpoint: %s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Errorf("read native request: %v", err)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		stream(w, r)
	}))
	t.Cleanup(server.Close)
	port, err := strconv.Atoi(strings.TrimPrefix(server.URL, "http://127.0.0.1:"))
	if err != nil {
		t.Fatal(err)
	}
	return &llamaServerRunner{port: port, cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1), options: api.Options{Runner: api.Runner{NumCtx: 2048}}}
}

func toolProgressEvent(t *testing.T, arguments string) string {
	t.Helper()
	delta := llamaServerChatToolCall{Index: 0, ID: "call_write"}
	delta.Function.Name = "write_file"
	delta.Function.Arguments = arguments
	return toolProgressDeltaEvent(t, delta)
}

func toolProgressDeltaEvent(t *testing.T, delta llamaServerChatToolCall) string {
	t.Helper()
	event, err := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []llamaServerChatToolCall{delta}}}}})
	if err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf("data: %s\n\n", event)
}

func TestLlamaServerChatToolProgressBeforeCompletion(t *testing.T) {
	first := toolProgressEvent(t, `{"content":"`)
	// Continuation fragments need not repeat the call identity.
	lastDelta := llamaServerChatToolCall{Index: 0}
	lastDelta.Function.Arguments = "ação\\ntexto\"}"
	last := toolProgressDeltaEvent(t, lastDelta)
	finish := "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n"
	release := make(chan struct{})
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	runner := toolProgressTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, first)
		w.(http.Flusher).Flush()
		select {
		case <-release:
			fmt.Fprint(w, last, finish)
		case <-r.Context().Done():
		}
	})
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	responses := make(chan ChatResponse, 8)
	done := make(chan error, 1)
	go func() { done <- runner.Chat(ctx, ChatRequest{}, func(r ChatResponse) { responses <- r }) }()
	select {
	case r := <-responses:
		if r.GenerationProgress == nil || r.GenerationProgress.Phase != "tool_call" || r.Done || r.Message.Content != "" || r.Message.Thinking != "" || len(r.Message.ToolCalls) != 0 {
			t.Fatalf("progress before completion = %+v", r)
		}
	case err := <-done:
		t.Fatalf("generation ended before progress: %v", err)
	case <-ctx.Done():
		t.Fatal("native tool fragments did not produce immediate progress")
	}
	// A quiet upstream must remain quiet: progress comes only from real
	// fragments, rather than an independent heartbeat masking a stalled tool.
	select {
	case r := <-responses:
		t.Fatalf("idle native stream manufactured progress: %+v", r)
	case err := <-done:
		t.Fatalf("generation ended before final fragment: %v", err)
	case <-time.After(1100 * time.Millisecond):
	}
	close(release)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal("tool completion did not finish")
	}
	var final ChatResponse
	for len(responses) > 0 {
		r := <-responses
		if r.GenerationProgress != nil {
			if r.Done || len(r.Message.ToolCalls) != 0 {
				t.Fatalf("partial tool escaped progress: %+v", r)
			}
			continue
		}
		if final.Done || !r.Done {
			t.Fatalf("unexpected response before final tool batch: %+v", r)
		}
		final = r
	}
	if !final.Done || len(final.Message.ToolCalls) != 1 || final.GenerationProgress != nil {
		t.Fatalf("final response = %+v", final)
	}
	call := final.Message.ToolCalls[0]
	content, _ := call.Function.Arguments.Get("content")
	if call.ID != "call_write" || call.Function.Name != "write_file" || content != "ação\ntexto" {
		t.Fatalf("tool was changed during progress: %+v", call)
	}
}

func TestLlamaServerChatToolProgressRequiresRealFragments(t *testing.T) {
	finish := "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	valid := toolProgressEvent(t, `{}`)
	for _, tc := range []struct {
		name, stream string
		progress     int
		wantError    bool
	}{
		{"empty tool delta", "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0}]}}]}\n\n" + finish, 0, true},
		{"visible output already signals activity", "data: {\"choices\":[{\"delta\":{\"content\":\"ready\",\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"function\":{\"name\":\"write\",\"arguments\":\"{}\"}}]}}]}\n\n" + finish, 0, false},
		{"terminal delta needs no progress", "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"function\":{\"name\":\"write\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", 0, false},
		{"negative index", "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":-1,\"id\":\"bad\"}]}}]}\n\n", 0, true},
		{"malformed frame", "data: not-json\n\n", 0, true},
		{"upstream disconnect", "", 0, true},
		{"malformed after real activity", valid + "data: not-json\n\n", 1, true},
		{"disconnect after real activity", valid, 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runner, _, _ := terminalTestRunner(t, tc.stream, nil)
			progress := 0
			err := runner.Chat(t.Context(), ChatRequest{}, func(r ChatResponse) {
				if r.GenerationProgress != nil {
					progress++
					if r.GenerationProgress.Phase != "tool_call" || r.Done || len(r.Message.ToolCalls) != 0 {
						t.Errorf("invalid progress: %+v", r)
					}
				}
				if tc.wantError && r.Done {
					t.Error("incomplete or invalid native response declared done")
				}
			})
			if (err != nil) != tc.wantError || progress != tc.progress {
				t.Fatalf("progress=%d error=%v; want progress=%d error=%v", progress, err, tc.progress, tc.wantError)
			}
		})
	}
}

func TestLlamaServerChatToolProgressStopsOnCancellation(t *testing.T) {
	first := toolProgressEvent(t, `{"content":"`)
	release := make(chan struct{})
	defer close(release)
	runner := toolProgressTestRunner(t, func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, first)
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-release:
		}
	})
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	responses := make(chan ChatResponse, 8)
	done := make(chan error, 1)
	go func() { done <- runner.Chat(ctx, ChatRequest{}, func(r ChatResponse) { responses <- r }) }()
	select {
	case r := <-responses:
		if r.GenerationProgress == nil || r.Done || len(r.Message.ToolCalls) != 0 {
			t.Fatalf("expected initial tool progress: %+v", r)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no tool progress before cancellation")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Chat error = %v, want cancellation", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("canceled stream did not return")
	}
	if len(responses) != 0 {
		t.Fatal("canceled tool generation emitted another callback")
	}
}
