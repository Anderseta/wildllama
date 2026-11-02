package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

var completionFieldEvents = []string{
	`{"content":"","stop":false,"prompt_progress":{"total":100,"cache":60,"processed":80,"time_ms":10},"index":0,"id_slot":1}`,
	`{"content":"<think>raciocínio</think>Olá","stop":false,"completion_probabilities":[{"token":"Olá","logprob":-0.25,"prob":0.7,"top_logprobs":[{"token":"Oi","logprob":-1.5,"prob":0.2}]}],"tokens":[42],"model":"fixture"}`,
	`{"content":"fim","stop":true,"stop_type":"limit","timings":{"cache_n":60,"prompt_n":40,"prompt_ms":25.5,"predicted_n":2,"predicted_ms":10.25,"draft_n":6,"draft_n_accepted":1,"gpu_first":{"cpu_heavy_ops_submitted":0,"gpu_heavy_ops_submitted":42}},"prompt":"unused","generation_settings":{"grammar":"unused"}}`,
}

func filterCompletionFields(data []byte, fields []string) ([]byte, error) {
	var full map[string]json.RawMessage
	if err := json.Unmarshal(data, &full); err != nil {
		return nil, err
	}
	filtered := make(map[string]json.RawMessage, len(fields))
	for _, field := range fields {
		if value, exists := full[field]; exists {
			filtered[field] = value
		}
	}
	return json.Marshal(filtered)
}

func TestCompletionResponseFieldsPreserveEveryConsumer(t *testing.T) {
	request := llamaServerCompletionRequest{ResponseFields: []string{"content", "stop", "stop_type", "timings", "completion_probabilities", "prompt_progress"}}
	selected := make(map[string]bool)
	for _, field := range request.ResponseFields {
		selected[field] = true
	}
	responseType := reflect.TypeOf(llamaServerCompletionResponse{})
	for i := range responseType.NumField() {
		key := strings.Split(responseType.Field(i).Tag.Get("json"), ",")[0]
		if !selected[key] {
			t.Errorf("native field selection omits consumed response field %q", key)
		}
	}
	for _, event := range completionFieldEvents {
		filtered, err := filterCompletionFields([]byte(event), request.ResponseFields)
		if err != nil {
			t.Fatal(err)
		}
		var full, reduced llamaServerCompletionResponse
		if err := json.Unmarshal([]byte(event), &full); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(filtered, &reduced); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(full, reduced) {
			t.Fatalf("response field selection changed consumed data:\nfull=%+v\nfiltered=%+v", full, reduced)
		}
	}
}

func TestCompletionRequestsOnlyConsumedNativeFields(t *testing.T) {
	var reference []CompletionResponse
	for _, filtered := range []bool{false, true} {
		t.Run(fmt.Sprintf("filtered_%t", filtered), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					fmt.Fprint(w, `{"status":"ok"}`)
					return
				}
				var request llamaServerCompletionRequest
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
					return
				}
				if !reflect.DeepEqual(request.ResponseFields, []string{"content", "stop", "stop_type", "timings", "completion_probabilities", "prompt_progress"}) {
					t.Errorf("unexpected native field selection: %v", request.ResponseFields)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				for _, original := range completionFieldEvents {
					event := []byte(original)
					if filtered {
						var err error
						event, err = filterCompletionFields(event, request.ResponseFields)
						if err != nil {
							t.Error(err)
							return
						}
					}
					fmt.Fprintf(w, "data: %s\n\n", event)
				}
			}))
			defer server.Close()
			u, _ := url.Parse(server.URL)
			port, _ := strconv.Atoi(u.Port())
			runner := &llamaServerRunner{port: port, cmd: fakeRunningCmd(), sem: semaphore.NewWeighted(1), options: api.Options{Runner: api.Runner{NumCtx: 2048}}}
			opts := api.DefaultOptions()
			var responses []CompletionResponse
			if err := runner.Completion(t.Context(), CompletionRequest{Prompt: "fixture", Options: &opts, Logprobs: true, TopLogprobs: 1}, func(response CompletionResponse) {
				responses = append(responses, response)
			}); err != nil {
				t.Fatal(err)
			}
			if !filtered {
				reference = responses
			} else if !reflect.DeepEqual(reference, responses) {
				t.Fatalf("filtered events changed callbacks:\nfull=%+v\nfiltered=%+v", reference, responses)
			}
		})
	}
}
