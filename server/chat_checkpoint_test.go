package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

func TestChatCheckpointRequiresLocalTextOwner(t *testing.T) {
	for _, tc := range []struct{ peer, owner, want string }{
		{"127.0.0.1:5000", "project-1", "project-1"},
		{"[::1]:5000", "project-1", "project-1"},
		{"192.0.2.1:5000", "project-1", ""},
		{"garbage", "project-1", ""},
		{"127.0.0.1:5000", "bad/id", ""},
	} {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodPost, "/api/chat", nil)
		c.Request.RemoteAddr = tc.peer
		c.Request.Header.Set("X-Forwarded-For", "127.0.0.1")
		c.Request.Header.Set("X-WildLlama-Chat-Session", tc.owner)
		if got := localCheckpointSession(c, nil); got != tc.want {
			t.Errorf("unexpected ownership for %s / %s: %q", tc.peer, tc.owner, got)
		}
		if got := localCheckpointSession(c, []api.Message{{Images: []api.ImageData{{1}}}}); got != "" {
			t.Errorf("image request opted in: %q", got)
		}
	}
}
