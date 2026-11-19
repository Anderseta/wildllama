package server

import (
	"net"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// Only a client on the same host can opt a conversation into local storage.
// Use the transport peer, never proxy-supplied forwarding headers.
func localCheckpointSession(c *gin.Context, messages []api.Message) string {
	id := c.GetHeader("X-WildLlama-Chat-Session")
	if !llm.ValidCheckpointSession(id) {
		return ""
	}
	host, _, err := net.SplitHostPort(c.Request.RemoteAddr)
	if err != nil || !net.ParseIP(host).IsLoopback() || countChatImages(messages) != 0 {
		return ""
	}
	return id
}
