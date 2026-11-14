package server

import "github.com/gin-gonic/gin"

// Immutable per-output facts, independent of the bounded recent-job history.
// Kept out of imageAsset so polling and image galleries don't resend prompts.
type imageGenerationMetadata struct {
	Prompt         string  `json:"prompt"`
	NegativePrompt string  `json:"negative_prompt"`
	CFGScale       float64 `json:"cfg_scale"`
	Steps          int     `json:"steps"`
	Sampler        string  `json:"sampler"`
	Scheduler      string  `json:"scheduler,omitempty"`
	ProfileID      string  `json:"profile_id"`
	ProfileName    string  `json:"profile_name,omitempty"`
	ModelID        string  `json:"model_id"`
}

func imageMetadataFor(g *imageGenerationSummary) *imageGenerationMetadata {
	if g == nil {
		return nil
	}
	return &imageGenerationMetadata{Prompt: g.Prompt, NegativePrompt: g.NegativePrompt,
		CFGScale: g.Profile.CFGScale, Steps: g.Profile.Steps, Sampler: g.Profile.Sampler,
		Scheduler: g.Profile.Scheduler, ProfileID: g.ProfileID, ProfileName: g.Profile.Name, ModelID: g.Profile.ModelID}
}

// Upgrade surviving legacy outputs once, before recent jobs can be pruned.
// Never infer old prompts or parameters from a profile edited later.
func (v *imageService) backfillImageMetadata() error {
	byJob := make(map[string]*imageGenerationSummary, len(v.jobs))
	for _, job := range v.jobs {
		byJob[job.ID] = job.Generation
	}
	changed := map[string]imageAssetRecord{}
	for id, asset := range v.assets {
		if asset.Input || asset.DeletionCompleted || asset.Generation != nil {
			continue
		}
		if metadata := imageMetadataFor(byJob[asset.JobID]); metadata != nil {
			asset.Generation = metadata
			changed[id] = asset
		}
	}
	if len(changed) == 0 {
		return nil
	}
	// The JSON-to-SQLite migration snapshots v.assets; populate it first.
	for id, asset := range changed {
		v.assets[id] = asset
	}
	if err := v.persistImageAssetsLocked(changed, nil); err != nil {
		for id, asset := range changed {
			asset.Generation = nil
			v.assets[id] = asset
		}
		return err
	}
	return nil
}

func (s *Server) imageAssetMetadataHandler(c *gin.Context) {
	v := s.imageService()
	v.mu.Lock()
	asset, ok := v.assets[c.Param("id")+"/"+c.Param("filename")]
	deleted := v.deletedSessions[asset.SessionID] || v.deletedRequests[asset.RequestID]
	v.mu.Unlock()
	if !ok || deleted || asset.DeletionCompleted || asset.Input || (c.Query("session_id") != "" && c.Query("session_id") != asset.SessionID) {
		c.JSON(404, gin.H{"error": "image metadata not found"})
		return
	}
	c.Header("Cache-Control", "no-store")
	c.JSON(200, gin.H{"asset": asset.imageAsset, "generation": asset.Generation})
}
