package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"

	"github.com/gin-gonic/gin"
)

func validImageDeletionAsset(asset imageAssetRecord) bool {
	name := asset.Filename == "image.png" || !asset.Input && imageBatchOutputName(asset.Filename) ||
		asset.Input && (asset.Filename == "image.jpg" || asset.Filename == "image.webp")
	return imageSafeID(asset.JobID) && (asset.RequestID == "" || imageSafeID(asset.RequestID)) && name && asset.AssetID == asset.JobID+"/"+asset.Filename &&
		filepath.IsAbs(asset.OutputRoot) && filepath.Clean(asset.Path) == filepath.Clean(filepath.Join(asset.OutputRoot, asset.relativePath()))
}

// Keep a durable fence independently of the bounded task history. Cleanup can
// be retried after an interrupted filesystem/SQLite write or a process restart.
func (v *imageService) loadImageAssetDeletions() {
	v.assetDeletions = map[string]imageAssetRecord{}
	f, err := os.Open(filepath.Join(v.root, "deleted-assets.json"))
	if errors.Is(err, os.ErrNotExist) {
		return
	}
	if err != nil {
		v.assetDeleteErr = err
		return
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, (16<<20)+1))
	if err != nil || len(data) > 16<<20 || json.Unmarshal(data, &v.assetDeletions) != nil || v.assetDeletions == nil {
		v.assetDeleteErr = errors.New("invalid image asset deletion journal")
		return
	}
	for id, asset := range v.assetDeletions {
		if id != asset.AssetID || !validImageDeletionAsset(asset) {
			v.assetDeleteErr = errors.New("invalid image asset deletion identity")
			return
		}
	}
	if len(v.assetDeletions) > 0 {
		v.assetDeleteErr = v.fenceImageAssetRequestsLocked()
		if v.assetDeleteErr == nil {
			v.assetDeleteErr = v.cleanupImageAssetDeletionsLocked()
		}
	}
}

func (v *imageService) fenceImageAssetRequestsLocked() error {
	if v.deletedRequests == nil {
		v.deletedRequests = map[string]bool{}
	}
	if v.deletedSessions == nil {
		v.deletedSessions = map[string]bool{}
	}
	changed := false
	for _, asset := range v.assetDeletions {
		if asset.RequestID != "" && !v.deletedRequests[asset.RequestID] {
			v.deletedRequests[asset.RequestID] = true
			changed = true
		}
	}
	if changed {
		return v.persistImageDeletionsLocked()
	}
	return nil
}

func (v *imageService) cleanupImageAssetDeletionsLocked() error {
	if v.ownedDirectories == nil {
		v.ownedDirectories = map[string]imageOwnedDirectory{}
	}
	dirs := map[string]imageAssetRecord{}
	pending := []string{}
	for _, asset := range v.assetDeletions {
		if !validImageDeletionAsset(asset) {
			return errors.New("invalid image asset deletion identity")
		}
	}
	for id, asset := range v.assetDeletions {
		delete(v.assets, id)
		if asset.DeletionCompleted {
			continue
		}
		pending = append(pending, id)
		if err := removeImageOwnedPath(asset.OutputRoot, asset.relativePath(), false); err != nil {
			return err
		}
		dirs[asset.OutputRoot+"\x00"+filepath.Dir(asset.relativePath())] = asset
	}
	for _, asset := range dirs {
		relative := filepath.Dir(asset.relativePath())
		shared := false
		for _, other := range v.assets {
			if filepath.Clean(other.OutputRoot) == filepath.Clean(asset.OutputRoot) && filepath.Dir(other.relativePath()) == relative {
				shared = true
				break
			}
		}
		if !shared {
			if err := removeImageOwnedPath(asset.OutputRoot, relative, true); err != nil {
				return err
			}
			delete(v.ownedDirectories, asset.JobID)
		}
	}
	for _, job := range v.jobs {
		touched := false
		for _, asset := range v.assetDeletions {
			if asset.JobID == job.ID {
				touched = true
				break
			}
		}
		if !touched {
			continue
		}
		for id, asset := range v.assetDeletions {
			if asset.JobID == job.ID && !slices.Contains(job.DeletedAssetIDs, id) {
				job.DeletedAssetIDs = append(job.DeletedAssetIDs, id)
			}
		}
		results := slices.DeleteFunc(append([]imageAsset(nil), job.Results...), func(a imageAsset) bool { _, gone := v.assetDeletions[a.AssetID]; return gone })
		if len(job.Results) == 0 && job.Result != nil {
			if _, gone := v.assetDeletions[job.Result.AssetID]; !gone {
				results = append(results, *job.Result)
			}
		}
		job.Results, job.Result = results, nil
		if len(results) > 0 {
			first := results[0]
			job.Result = &first
		}
	}
	if err := v.persistImageAssetsLocked(nil, pending); err != nil {
		return err
	}
	if err := writeImageJSON(filepath.Join(v.root, "session-directories.json"), v.ownedDirectories); err != nil {
		return err
	}
	if err := v.persistJobsLocked(); err != nil {
		return err
	}
	for _, id := range pending {
		asset := v.assetDeletions[id]
		if err := removeImageOwnedPath(v.root, filepath.Join("job-state", asset.JobID+".json"), false); err != nil {
			return err
		}
	}
	if len(pending) > 0 {
		for _, id := range pending {
			asset := v.assetDeletions[id]
			asset.DeletionCompleted = true
			v.assetDeletions[id] = asset
		}
		if err := writeImageJSON(filepath.Join(v.root, "deleted-assets.json"), v.assetDeletions); err != nil {
			for _, id := range pending {
				asset := v.assetDeletions[id]
				asset.DeletionCompleted = false
				v.assetDeletions[id] = asset
			}
			return err
		}
	}
	return nil
}

func (s *Server) imageAssetDeleteHandler(c *gin.Context) {
	id := c.Param("id") + "/" + c.Param("filename")
	if !imageSafeID(c.Param("id")) {
		c.JSON(400, gin.H{"error": "invalid image identity"})
		return
	}
	var request struct {
		SessionID string `json:"session_id"`
	}
	if c.Request.ContentLength != 0 && !imageBind(c, &request) {
		return
	}
	v := s.imageService()
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.assetDeleteErr != nil {
		// A recorded filesystem failure is retryable; malformed journals are not.
		if len(v.assetDeletions) == 0 {
			c.JSON(500, gin.H{"error": v.assetDeleteErr.Error()})
			return
		}
		for key, entry := range v.assetDeletions {
			if key != entry.AssetID || !validImageDeletionAsset(entry) {
				c.JSON(500, gin.H{"error": "invalid image deletion journal"})
				return
			}
		}
	}
	asset, exists := v.assets[id]
	if !exists {
		asset, exists = v.assetDeletions[id]
	}
	if !exists {
		c.JSON(404, gin.H{"error": "image asset not found"})
		return
	}
	if request.SessionID != "" && asset.SessionID != "" && request.SessionID != asset.SessionID {
		c.JSON(403, gin.H{"error": "image belongs to another chat"})
		return
	}
	if !validImageDeletionAsset(asset) {
		c.JSON(400, gin.H{"error": "invalid managed image identity"})
		return
	}
	for _, job := range v.jobs {
		if (!imageTerminal(job.Status) || job.executing) && (job.ID == asset.JobID || slices.Contains(imageJobReferences(job), id)) {
			c.JSON(409, gin.H{"error": "A imagem está em uso. Aguarde a geração terminar ou cancele-a antes de excluir."})
			return
		}
	}
	if v.assetDeletions == nil {
		v.assetDeletions = map[string]imageAssetRecord{}
	}
	v.assetDeletions[id] = asset
	removedSet := map[string]bool{id: true}
	// Collect managed input copies only when no remaining image or other job
	// needs them. Never traverse to source files outside the registered store.
	refs := append(append([]string(nil), asset.ReferenceAssetIDs...), asset.ParentAssetID)
	for _, job := range v.jobs {
		if job.ID == asset.JobID {
			refs = append(refs, imageJobReferences(job)...)
		}
	}
	keepBatchReferences := false
	for key, other := range v.assets {
		if _, gone := v.assetDeletions[key]; !gone && other.JobID == asset.JobID && !other.Input {
			keepBatchReferences = true
			break
		}
	}
	for _, ref := range refs {
		if _, gone := v.assetDeletions[ref]; gone {
			removedSet[ref] = true
			continue
		}
		input, ok := v.assets[ref]
		if !ok || !input.Input {
			continue
		}
		used := false
		for key, other := range v.assets {
			if _, gone := v.assetDeletions[key]; gone {
				continue
			}
			if other.ParentAssetID == ref || slices.Contains(other.ReferenceAssetIDs, ref) {
				used = true
				break
			}
		}
		for _, job := range v.jobs {
			if job.ID == asset.JobID && !keepBatchReferences {
				continue
			}
			if slices.Contains(imageJobReferences(job), ref) {
				used = true
				break
			}
		}
		if !used {
			v.assetDeletions[ref] = input
			removedSet[ref] = true
		}
	}
	if err := writeImageJSON(filepath.Join(v.root, "deleted-assets.json"), v.assetDeletions); err != nil {
		// Reload before another deletion rather than acting on an uncommitted fence.
		v.assetDeleteErr = fmt.Errorf("cannot persist image deletion: %w", err)
		c.JSON(500, gin.H{"error": v.assetDeleteErr.Error()})
		return
	}
	if err := v.fenceImageAssetRequestsLocked(); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	if err := v.cleanupImageAssetDeletionsLocked(); err != nil {
		v.assetDeleteErr = err
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	v.assetDeleteErr = nil
	removed := make([]string, 0, len(removedSet))
	for key := range removedSet {
		removed = append(removed, key)
	}
	c.JSON(http.StatusOK, gin.H{"ok": true, "deleted_asset_ids": removed})
}
