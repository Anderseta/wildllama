package server

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

// A fresh identity/digest probe lets clients reuse a confirmed profile without
// enumerating every model or rediscovering the listener PID on every message.
// The optional model lookup reads metadata only; it never starts a runner.
func (s *Server) versionHandler(c *gin.Context) {
	s.identityOnce.Do(func() { s.instanceID = uuid.NewString() })
	protocol := s.protocolCapabilities()
	protocol["instance_id"] = s.instanceID
	protocol["features"].(gin.H)["profile_identity"] = true
	requested := c.Query("model")
	if header := c.GetHeader("X-Wildllama-Profile-Model"); header != "" {
		requested = header
	}
	if requested != "" || c.Request.URL.Query().Has("model") {
		check := gin.H{"model": requested}
		name := model.ParseName(requested)
		if len(requested) > 1024 || !name.IsValid() || !name.IsFullyQualified() {
			check["error"] = "invalid_model"
		} else if mf, err := manifest.ParseNamedManifest(name); err == nil {
			check["digest"] = mf.Digest()
		} else {
			check["error"] = "model_unavailable"
		}
		protocol["profile_check"] = check
	}
	c.JSON(http.StatusOK, gin.H{"version": version.Version, "capabilities": []string{"runner_env", "num_ubatch", "embedding_batch", "gpu_exclusive", "gpu_vram_limit_mib", "gpu_diagnostics", "live_activity", "moe_runtime", "kv_cache_initial_tokens", "reasoning_budget", "huggingface_catalog", "download_cleanup", "model_disk_headroom"}, "wildllama": protocol})
}
