package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestTerminalFrameDoesNotWaitForServerEOF(t *testing.T) {
	for _, chat := range []bool{false, true} {
		t.Run(strconv.FormatBool(chat), func(t *testing.T) {
			releaseServer := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					fmt.Fprint(w, `{"status":"ok"}`)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if chat {
					fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"finished\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
				} else {
					fmt.Fprint(w, "data: {\"content\":\"finished\",\"stop\":true,\"stop_type\":\"eos\"}\n\n")
				}
				w.(http.Flusher).Flush()
				select {
				case <-r.Context().Done():
				case <-releaseServer:
				}
			}))
			defer server.Close()
			defer close(releaseServer)
			u, _ := url.Parse(server.URL)
			port, _ := strconv.Atoi(u.Port())
			runner := &llamaServerRunner{port: port, sem: semaphore.NewWeighted(1), options: api.Options{Runner: api.Runner{NumCtx: 2048}}}
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
			defer cancel()
			finished := make(chan error, 1)
			done := false
			go func() {
				opts := api.DefaultOptions()
				if chat {
					finished <- runner.Chat(ctx, ChatRequest{Options: &opts}, func(response ChatResponse) { done = response.Done })
				} else {
					finished <- runner.Completion(ctx, CompletionRequest{Prompt: "input", Options: &opts}, func(response CompletionResponse) { done = response.Done })
				}
			}()
			select {
			case err := <-finished:
				if err != nil || !done {
					t.Fatalf("terminal result: done=%t error=%v", done, err)
				}
			case <-time.After(time.Second):
				cancel()
				<-finished
				t.Fatal("completed response waited for EOF")
			}
			if !runner.sem.TryAcquire(1) {
				t.Fatal("terminal response did not release runner slot")
			}
			runner.sem.Release(1)
		})
	}
}
