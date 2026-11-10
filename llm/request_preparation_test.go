package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

type preparationTestMarshaler struct{ calls atomic.Int32 }

func (s *preparationTestMarshaler) MarshalJSON() ([]byte, error) {
	s.calls.Add(1)
	return []byte(`{"valid":true}`), nil
}

func TestNativeCancelledRequestSkipsSerialization(t *testing.T) {
	for _, route := range []string{"batch", "tokenize", "template"} {
		t.Run(route, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			var spy preparationTestMarshaler
			runner := &llamaServerRunner{}
			var err error
			switch route {
			case "batch":
				var output any
				err = runner.batchJSON(ctx, "/tokenize", &spy, &output)
			case "tokenize":
				_, err = runner.tokenize(ctx, &spy, false)
			case "template":
				arguments := api.NewToolCallFunctionArguments()
				arguments.Set("data", &spy)
				_, err = runner.applyChatTemplate(ctx, ChatRequest{Messages: []api.Message{{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "write", Arguments: arguments}}}}}}, false)
			}
			if !errors.Is(err, context.Canceled) || spy.calls.Load() != 0 {
				t.Fatalf("cancelled request was encoded: calls=%d error=%v", spy.calls.Load(), err)
			}
		})
	}
}

func TestSystemOnePreparationWaitsForAdmission(t *testing.T) {
	sem := semaphore.NewWeighted(1)
	if err := sem.Acquire(t.Context(), 1); err != nil {
		t.Fatal(err)
	}
	defer sem.Release(1)
	runner := &llamaServerRunner{sem: sem}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		// Invalid JSON is a sentinel for reaching the serialization stage; it
		// must remain untouched while this request waits for an inference slot.
		_, _, err := runner.SystemOne(ctx, json.RawMessage(`{"unfinished":`), json.RawMessage(`{}`), nil)
		done <- err
	}()
	select {
	case err := <-done:
		t.Fatalf("queued request reached serialization: %v", err)
	case <-time.After(40 * time.Millisecond):
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled queued preparation returned %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancelled preparation kept waiting for the model slot")
	}
}

func TestSystemOneAdmissionPreservesPayloadAndReleasesSlot(t *testing.T) {
	state := json.RawMessage(`{"text":"ação com <tags> e dados"}`)
	questions := json.RawMessage(`{"q":{"type":"boolean","instructions":"validar"}}`)
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		var body struct {
			State, Questions json.RawMessage
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		var decodedState struct{ Text string }
		if err := json.Unmarshal(body.State, &decodedState); err != nil {
			t.Error(err)
			return
		}
		if r.URL.Path != "/v1/systemone" || decodedState.Text != "ação com <tags> e dados" || string(body.Questions) != string(questions) {
			t.Errorf("decision request changed: path=%s state=%s questions=%s", r.URL.Path, body.State, body.Questions)
		}
		fmt.Fprint(w, `{"answers":{"q":true},"usage":{"prompt_tokens":7}}`)
	}))
	defer server.Close()
	address, _ := url.Parse(server.URL)
	port, _ := strconv.Atoi(address.Port())
	sem := semaphore.NewWeighted(1)
	runner := &llamaServerRunner{port: port, sem: sem}
	answers, usage, err := runner.SystemOne(t.Context(), state, questions, nil)
	if err != nil || string(answers) != `{"q":true}` || string(usage) != `{"prompt_tokens":7}` || requests.Load() != 1 {
		t.Fatalf("decision response changed: answers=%s usage=%s error=%v", answers, usage, err)
	}
	if !sem.TryAcquire(1) {
		t.Fatal("successful decision leaked its inference slot")
	}
	sem.Release(1)
	if _, _, err := runner.SystemOne(t.Context(), json.RawMessage(`invalid`), questions, nil); err == nil {
		t.Fatal("invalid state was accepted")
	}
	if !sem.TryAcquire(1) {
		t.Fatal("failed serialization leaked its inference slot")
	}
	sem.Release(1)
}

func TestSystemOnePreparationReleasesAfterNativeFailure(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		status     int
		wantStatus int
	}{
		{"context rejection", `{"error":{"message":"increase the physical batch size"}}`, http.StatusInternalServerError, http.StatusBadRequest},
		{"native unavailable", `{"error":"native unavailable"}`, http.StatusServiceUnavailable, http.StatusServiceUnavailable},
		{"malformed response", `not json`, http.StatusOK, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				fmt.Fprint(w, tc.body)
			}))
			defer server.Close()
			address, _ := url.Parse(server.URL)
			port, _ := strconv.Atoi(address.Port())
			sem := semaphore.NewWeighted(1)
			runner := &llamaServerRunner{port: port, sem: sem}
			_, _, err := runner.SystemOne(t.Context(), json.RawMessage(`{}`), json.RawMessage(`{}`), nil)
			if err == nil {
				t.Fatal("native failure was hidden")
			}
			if tc.wantStatus != 0 {
				var status api.StatusError
				if !errors.As(err, &status) || status.StatusCode != tc.wantStatus {
					t.Fatalf("native status changed: %T %v", err, err)
				}
			}
			if !sem.TryAcquire(1) {
				t.Fatal("native failure leaked the inference slot")
			}
			sem.Release(1)
		})
	}
}

func TestSystemOnePreparationCancellationDuringNativeWork(t *testing.T) {
	started := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		close(started)
		<-r.Context().Done()
	}))
	defer server.Close()
	address, _ := url.Parse(server.URL)
	port, _ := strconv.Atoi(address.Port())
	sem := semaphore.NewWeighted(1)
	runner := &llamaServerRunner{port: port, sem: sem}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, _, err := runner.SystemOne(ctx, json.RawMessage(`{}`), json.RawMessage(`{}`), nil)
		done <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("native request did not start")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("native cancellation changed: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("native cancellation retained the inference slot")
	}
	if !sem.TryAcquire(1) {
		t.Fatal("cancelled native request leaked the inference slot")
	}
	sem.Release(1)
}

func BenchmarkCancelledNativePreparation(b *testing.B) {
	payload := strings.Repeat("x", 2<<20)
	state := json.RawMessage(`{"value":"` + payload + `"}`)
	ctx, cancel := context.WithCancel(b.Context())
	cancel()
	for _, route := range []string{"decision", "tokenize", "template"} {
		b.Run(route, func(b *testing.B) {
			runner := &llamaServerRunner{sem: semaphore.NewWeighted(1)}
			b.ReportAllocs()
			for b.Loop() {
				var err error
				switch route {
				case "decision":
					_, _, err = runner.SystemOne(ctx, state, json.RawMessage(`{}`), nil)
				case "tokenize":
					_, err = runner.Tokenize(ctx, payload)
				case "template":
					_, _, err = runner.ApplyChatTemplateWithTokenCount(ctx, ChatRequest{Messages: []api.Message{{Role: "user", Content: payload}}})
				}
				if !errors.Is(err, context.Canceled) {
					b.Fatal(err)
				}
			}
		})
	}
}
