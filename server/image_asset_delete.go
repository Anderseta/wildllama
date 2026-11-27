package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"slices"

	"github.com/gin-gonic/gin"
)

var (
	errImageAssetInUse    = errors.New("Uma das imagens está em uso. Aguarde a geração terminar antes de excluir.")
	errImageAssetIdentity = errors.New("invalid managed image identity")
)

func validImageDeletionAsset(asset imageAssetRecord) bool {
	name := validImageAssetFilename(asset.Filename, asset.Input)
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
	occupied := make(map[string]bool, len(v.assets))
	for _, asset := range v.assets {
		occupied[filepath.Clean(asset.OutputRoot)+"\x00"+filepath.Dir(asset.relativePath())] = true
	}
	for _, asset := range dirs {
		relative := filepath.Dir(asset.relativePath())
		if !occupied[filepath.Clean(asset.OutputRoot)+"\x00"+relative] {
			if err := removeImageOwnedPath(asset.OutputRoot, relative, true); err != nil {
				return err
			}
			delete(v.ownedDirectories, asset.JobID)
		}
	}
	byJob := map[string][]string{}
	for id, asset := range v.assetDeletions {
		byJob[asset.JobID] = append(byJob[asset.JobID], id)
	}
	for _, job := range v.jobs {
		ids := byJob[job.ID]
		if len(ids) == 0 {
			continue
		}
		for _, id := range ids {
			if !slices.Contains(job.DeletedAssetIDs, id) {
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
	removed, err := v.deleteImageAssetsLocked([]imageAssetRecord{asset}, "")
	imageAssetDeletionResponse(c, removed, err)
}

func imageAssetDeletionResponse(c *gin.Context, removed []string, err error) {
	if err == nil {
		c.JSON(http.StatusOK, gin.H{"ok": true, "deleted_asset_ids": removed})
		return
	}
	status := http.StatusInternalServerError
	switch {
	case errors.Is(err, errImageAssetInUse):
		status = http.StatusConflict
	case errors.Is(err, errImageAssetIdentity):
		status = http.StatusBadRequest
	}
	c.JSON(status, gin.H{"error": err.Error()})
}

func (v *imageService) deleteImageAssetsLocked(assets []imageAssetRecord, batch string) ([]string, error) {
	if v.assetDeleteErr != nil {
		// A recorded filesystem failure is retryable. Invalid journals are not.
		if len(v.assetDeletions) == 0 {
			return nil, v.assetDeleteErr
		}
		for key, entry := range v.assetDeletions {
			if key != entry.AssetID || !validImageDeletionAsset(entry) {
				return nil, errors.New("invalid image deletion journal")
			}
		}
	}
	journal := maps.Clone(v.assetDeletions)
	if journal == nil {
		journal = map[string]imageAssetRecord{}
	}
	removed := map[string]bool{}
	selectedJobs := map[string]bool{}
	var refs []string
	for _, asset := range assets {
		if !validImageDeletionAsset(asset) {
			return nil, errImageAssetIdentity
		}
		if _, exists := journal[asset.AssetID]; !exists {
			asset.DeletionBatch = batch
			journal[asset.AssetID] = asset
		}
		removed[asset.AssetID] = true
		selectedJobs[asset.JobID] = true
		refs = append(refs, asset.ReferenceAssetIDs...)
		refs = append(refs, asset.ParentAssetID)
	}
	retainedJobs := map[string]bool{}
	used := map[string]bool{}
	for id, asset := range v.assets {
		if _, gone := journal[id]; gone {
			continue
		}
		if !asset.Input {
			retainedJobs[asset.JobID] = true
		}
		used[asset.ParentAssetID] = true
		for _, ref := range asset.ReferenceAssetIDs {
			used[ref] = true
		}
	}
	// Collect managed input copies only when no remaining image or other job
	// needs them. Never traverse to source files outside the registered store.
	for _, job := range v.jobs {
		jobRefs := imageJobReferences(job)
		if !imageTerminal(job.Status) || job.executing {
			if selectedJobs[job.ID] {
				return nil, errImageAssetInUse
			}
			for _, ref := range jobRefs {
				if removed[ref] {
					return nil, errImageAssetInUse
				}
			}
		}
		if selectedJobs[job.ID] {
			refs = append(refs, jobRefs...)
			if !retainedJobs[job.ID] {
				continue
			}
		}
		for _, ref := range jobRefs {
			used[ref] = true
		}
	}
	for _, ref := range refs {
		if _, gone := journal[ref]; gone {
			removed[ref] = true
			continue
		}
		input, ok := v.assets[ref]
		if ok && input.Input && !used[ref] {
			if !validImageDeletionAsset(input) {
				return nil, errImageAssetIdentity
			}
			input.DeletionBatch = batch
			journal[ref] = input
			removed[ref] = true
		}
	}
	if err := writeImageJSON(filepath.Join(v.root, "deleted-assets.json"), journal); err != nil {
		return nil, fmt.Errorf("cannot persist image deletion: %w", err)
	}
	v.assetDeletions = journal
	if err := v.finishImageAssetDeletionLocked(); err != nil {
		return nil, err
	}
	ids := slices.Sorted(maps.Keys(removed))
	return ids, nil
}

func (v *imageService) finishImageAssetDeletionLocked() error {
	v.assetDeleteErr = v.fenceImageAssetRequestsLocked()
	if v.assetDeleteErr == nil {
		v.assetDeleteErr = v.cleanupImageAssetDeletionsLocked()
	}
	return v.assetDeleteErr
}
