package llm

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestNativeRPCPreservesCorrelationWithoutMutatingRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Request-ID") != "generation-123" {
			t.Errorf("native request lost correlation: %q", r.Header.Get("X-Request-ID"))
		}
		w.WriteHeader(204)
	}))
	defer server.Close()
	request, _ := http.NewRequestWithContext(api.WithRequestID(t.Context(), "generation-123"), "GET", server.URL, nil)
	response, err := newLlamaServerHTTPClient().Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if request.Header.Get("X-Request-ID") != "" {
		t.Fatal("transport modified caller-owned headers")
	}
}
