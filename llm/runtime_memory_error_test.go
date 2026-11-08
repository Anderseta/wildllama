package llm

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestRuntimeMemoryErrorsAreFramedAndRequestScoped(t *testing.T) {
	status := NewStatusWriter(io.Discard)
	runner := &llamaServerRunner{status: status}
	before := status.memoryFailureVersion()
	_, _ = status.Write([]byte("ggml_cuda_memory_reservation: GPU 0 allocation de"))
	_, _ = status.Write([]byte("nied: request 2.00 MiB, driver free 0.00 MiB\n"))
	var failure api.StatusError
	if err := runner.runtimeStreamError("Compute error.", before, 500); !errors.As(err, &failure) || failure.StatusCode != 507 || !strings.Contains(failure.ErrorMessage, "2.00 MiB") {
		t.Fatalf("memory failure was hidden: %v", err)
	}
	current := status.memoryFailureVersion()
	for _, test := range []struct {
		message string
		status  int
	}{{"Compute error.", 422}, {"invalid grammar", 500}} {
		if err := runner.runtimeStreamError(test.message, current, 500); !errors.As(err, &failure) || failure.StatusCode != test.status {
			t.Fatalf("old OOM contaminated a new request: %v", err)
		}
	}
}

func TestLlamaServerMemoryErrorFramesAbortChatAndCompletion(t *testing.T) {
	for _, chat := range []bool{false, true} {
		t.Run(fmt.Sprint("chat=", chat), func(t *testing.T) {
			status := NewStatusWriter(io.Discard)
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					fmt.Fprint(w, `{"status":"ok"}`)
					return
				}
				calls++
				_, _ = status.Write([]byte("ggml_cuda_memory_reservation: GPU 0 allocation denied: request 2.00 MiB, driver free 0.00 MiB\n"))
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintln(w, `data: {"error":{"code":500,"message":"Compute error.","type":"server_error"}}`)
			}))
			defer srv.Close()
			var port int
			parts := strings.Split(srv.URL, ":")
			fmt.Sscanf(parts[len(parts)-1], "%d", &port)
			runner := &llamaServerRunner{port: port, cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1), status: status, options: api.DefaultOptions()}
			opts := api.DefaultOptions()
			var err error
			if chat {
				err = runner.Chat(t.Context(), ChatRequest{Messages: []api.Message{{Role: "user", Content: "test"}}, Options: &opts}, func(ChatResponse) { t.Error("error frame produced a successful response") })
			} else {
				err = runner.Completion(t.Context(), CompletionRequest{Prompt: "test", Options: &opts}, func(CompletionResponse) { t.Error("error frame produced a successful response") })
			}
			var failure api.StatusError
			if !errors.As(err, &failure) || failure.StatusCode != 507 || !IsOutOfMemory(err) || calls != 1 {
				t.Fatalf("unexpected completion: calls=%d error=%v", calls, err)
			}
		})
	}
}
