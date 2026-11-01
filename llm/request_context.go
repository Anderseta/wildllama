package llm

import (
	"net/http"

	"github.com/ollama/ollama/api"
)

type requestIDTransport struct{ base http.RoundTripper }

func (t requestIDTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	if id := api.RequestID(request.Context()); id != "" && request.Header.Get("X-Request-ID") == "" {
		request = request.Clone(request.Context())
		request.Header.Set("X-Request-ID", id)
	}
	return t.base.RoundTrip(request)
}
