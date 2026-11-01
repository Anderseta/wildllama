package server

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// Cleanup never queues a writer: normal pulls/imports keep shared access and
// chat/index inference does not touch this lock.
var downloadCleanupGate sync.RWMutex

func (s *Server) DiscardDownloadHandler(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 8192)
	var req api.DeleteRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid download cleanup request"})
		return
	}
	name := model.ParseName(req.Model)
	if strings.EqualFold(name.Host, "hf.co") {
		name.Host = "huggingface.co"
	}
	if !name.IsValid() || !name.IsFullyQualified() {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid model download name"})
		return
	}
	if !downloadCleanupGate.TryLock() {
		c.JSON(http.StatusConflict, gin.H{"error": "Aguarde o download ou a importação em andamento terminar."})
		return
	}
	defer downloadCleanupGate.Unlock()
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()
	var stopping bool
	blobDownloadManager.Range(func(_, value any) bool {
		select {
		case <-value.(*blobDownload).done:
			return true
		case <-ctx.Done():
			stopping = true
			return false
		}
	})
	if stopping {
		c.JSON(http.StatusConflict, gin.H{"error": "O download ainda está encerrando. Tente novamente em instantes."})
		return
	}
	result, err := manifest.DiscardDownload(name)
	removeGGUFMetadata(result.DiscardedDigests...)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error(), "removed_files": result.RemovedFiles})
		return
	}
	c.JSON(http.StatusOK, result)
}
