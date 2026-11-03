package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gin-gonic/gin"
)

func validImageSessionID(id string) bool {
	return strings.TrimSpace(id) != "" && len(id) <= 256 && utf8.ValidString(id) && !strings.ContainsAny(id, "\x00\r\n")
}

type imageDeletionJournal struct {
	Sessions map[string]bool `json:"sessions"`
	Requests map[string]bool `json:"requests"`
}

type imageOwnedDirectory struct {
	SessionID string `json:"session_id"`
	Root      string `json:"output_root"`
}

func (v *imageService) recordImageDirectoryLocked(job *imageJob) error {
	if !filepath.IsAbs(job.Config.OutputDir) {
		return errors.New("image output root must be absolute")
	}
	if v.ownedDirectories == nil {
		v.ownedDirectories = map[string]imageOwnedDirectory{}
	}
	v.ownedDirectories[job.ID] = imageOwnedDirectory{SessionID: job.SessionID, Root: job.Config.OutputDir}
	return writeImageJSON(filepath.Join(v.root, "session-directories.json"), v.ownedDirectories)
}

func (v *imageService) persistImageDeletionsLocked() error {
	return writeImageJSON(filepath.Join(v.root, "deleted-sessions.json"), imageDeletionJournal{Sessions: v.deletedSessions, Requests: v.deletedRequests})
}

// Tombstones survive restart independently of bounded job history. A submit
// already in flight must never recreate a deleted conversation's image data.
func (v *imageService) loadDeletedImageSessions() {
	v.deletedSessions = map[string]bool{}
	v.deletedRequests = map[string]bool{}
	file, err := os.Open(filepath.Join(v.root, "deleted-sessions.json"))
	if err == nil {
		defer file.Close()
		data, readErr := io.ReadAll(io.LimitReader(file, (16<<20)+1))
		var journal imageDeletionJournal
		if readErr != nil || len(data) > 16<<20 || json.Unmarshal(data, &journal) != nil || journal.Sessions == nil || journal.Requests == nil {
			err = errors.New("invalid saved image session deletion journal")
		} else {
			v.deletedSessions, v.deletedRequests = journal.Sessions, journal.Requests
		}
	} else if errors.Is(err, os.ErrNotExist) {
		err = nil
	}
	if v.deletedSessions == nil {
		v.deletedSessions = map[string]bool{}
	}
	if v.deletedRequests == nil {
		v.deletedRequests = map[string]bool{}
	}
	for id := range v.deletedRequests {
		if !imageSafeID(id) {
			err = errors.New("invalid saved image request deletion identity")
		}
	}
	for id := range v.deletedSessions {
		if !validImageSessionID(id) {
			err = errors.New("invalid saved image session deletion identity")
		}
	}
	v.sessionDeleteErr = err
	v.ownedDirectories = map[string]imageOwnedDirectory{}
	if data, readErr := os.ReadFile(filepath.Join(v.root, "session-directories.json")); readErr == nil {
		if len(data) > 16<<20 || json.Unmarshal(data, &v.ownedDirectories) != nil || v.ownedDirectories == nil {
			v.sessionDeleteErr = errors.Join(v.sessionDeleteErr, errors.New("invalid saved image directory ownership"))
		}
		for jobID, dir := range v.ownedDirectories {
			if !imageSafeID(jobID) || !validImageSessionID(dir.SessionID) || !filepath.IsAbs(dir.Root) {
				v.sessionDeleteErr = errors.Join(v.sessionDeleteErr, errors.New("invalid saved image directory identity"))
			}
		}
	} else if !errors.Is(readErr, os.ErrNotExist) {
		v.sessionDeleteErr = errors.Join(v.sessionDeleteErr, readErr)
	}
	for _, job := range v.jobs {
		if job.SessionID == "" {
			continue
		}
		for id, asset := range v.assets {
			if asset.JobID == job.ID && asset.SessionID == "" {
				asset.SessionID = job.SessionID
				if job.Generation != nil {
					asset.ReferenceAssetIDs = append([]string(nil), job.Generation.ReferenceAssetIDs...)
				}
				v.assets[id] = asset
				v.assetOwnersDirty = true
			}
		}
	}
}

func (v *imageService) checkImageSessionLocked(id string, requestIDs ...string) error {
	if v.sessionDeleteErr != nil {
		return fmt.Errorf("cannot verify image session deletion journal: %w", v.sessionDeleteErr)
	}
	for _, requestID := range requestIDs {
		if requestID != "" && v.deletedRequests[requestID] {
			return errors.New("this image request belongs to a deleted session")
		}
	}
	if id == "" { // Standalone image-tab jobs have no conversation lifecycle.
		return nil
	}
	if !validImageSessionID(id) {
		return errors.New("invalid image session_id")
	}
	if v.deletedSessions[id] {
		changed := false
		for _, requestID := range requestIDs {
			if imageSafeID(requestID) && !v.deletedRequests[requestID] {
				if v.deletedRequests == nil {
					v.deletedRequests = map[string]bool{}
				}
				v.deletedRequests[requestID] = true
				changed = true
			}
		}
		if changed {
			if err := v.persistImageDeletionsLocked(); err != nil {
				return fmt.Errorf("cannot persist deleted image request: %w", err)
			}
		}
		return errors.New("this image session was deleted")
	}
	return nil
}

func imageJobReferences(job *imageJob) []string {
	ids := append([]string{}, job.Request.ReferenceImages...)
	ids = append(ids, job.Request.InputAssetID, job.ParentAssetID, job.Request.ParentAssetID)
	if job.Generation != nil {
		ids = append(ids, job.Generation.ReferenceAssetIDs...)
	}
	var result []string
	for _, id := range ids {
		if id != "" {
			result = append(result, id)
		}
	}
	return result
}

func (s *Server) imageSessionDeleteHandler(c *gin.Context) {
	id := c.Param("session_id")
	if !validImageSessionID(id) {
		c.JSON(400, gin.H{"error": "a nonempty image session_id is required"})
		return
	}
	var request struct {
		LegacyAssetIDs []string `json:"legacy_asset_ids"`
	}
	if c.Request.ContentLength != 0 && !imageBind(c, &request) {
		return
	}
	if len(request.LegacyAssetIDs) > 4096 {
		c.JSON(400, gin.H{"error": "at most 4096 legacy image identities may be supplied per deletion"})
		return
	}
	v := s.imageService()
	v.mu.Lock()
	if v.sessionDeleteErr != nil {
		v.mu.Unlock()
		c.JSON(500, gin.H{"error": v.sessionDeleteErr.Error()})
		return
	}
	if v.deletedSessions == nil {
		v.deletedSessions = map[string]bool{}
	}
	if v.deletedRequests == nil {
		v.deletedRequests = map[string]bool{}
	}
	v.deletedSessions[id] = true
	for _, job := range v.jobs {
		if job.SessionID == id && imageSafeID(job.RequestID) {
			v.deletedRequests[job.RequestID] = true
		}
	}
	for _, asset := range v.assets {
		if asset.SessionID == id && imageSafeID(asset.RequestID) {
			v.deletedRequests[asset.RequestID] = true
		}
	}
	if err := v.persistImageDeletionsLocked(); err != nil {
		v.mu.Unlock()
		c.JSON(500, gin.H{"error": "cannot persist image deletion: " + err.Error()})
		return
	}
	for _, job := range v.jobs {
		if job.SessionID != id {
			continue
		}
		if job.cancel != nil {
			job.cancel()
		}
		if job.Status == "queued" {
			now := time.Now().UTC()
			job.Status, job.Phase, job.FinishedAt = "cancelled", "cancelled", &now
		}
	}
	v.compactImageQueueLocked()
	if err := v.persistJobsLocked(); err != nil {
		v.mu.Unlock()
		c.JSON(500, gin.H{"error": "cannot persist cancelled image jobs: " + err.Error()})
		return
	}
	v.mu.Unlock()

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()
	// The upload owns its staging files until it releases this slot. Its commit
	// rechecks the tombstone and removes bytes instead of publishing them.
	select {
	case v.uploadSlot <- struct{}{}:
		defer func() { <-v.uploadSlot }()
	case <-ctx.Done():
		c.JSON(409, gin.H{"error": "image uploads are still stopping; retry deleting the chat", "retryable": true})
		return
	}
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	for {
		v.mu.Lock()
		running := false
		for _, job := range v.jobs {
			if job.SessionID == id && job.executing {
				running = true
				break
			}
		}
		if !running {
			jobs, assets, retained, err := v.deleteImageSessionFilesLocked(id, request.LegacyAssetIDs)
			v.mu.Unlock()
			if err != nil {
				c.JSON(500, gin.H{"error": err.Error()})
				return
			}
			c.JSON(200, gin.H{"ok": true, "session_id": id, "removed_jobs": jobs, "removed_assets": assets, "retained_shared_assets": retained})
			return
		}
		v.mu.Unlock()
		select {
		case <-ctx.Done():
			c.JSON(409, gin.H{"error": "image jobs are still stopping; retry deleting the chat", "retryable": true})
			return
		case <-ticker.C:
		}
	}
}

// No client-supplied path is accepted. os.Root confines removal to the recorded
// managed output root; RemoveAll unlinks child symlinks without following them.
func removeImageOwnedPath(rootPath, relative string, recursive bool) error {
	if !filepath.IsLocal(relative) || relative == "." {
		return errors.New("invalid relative image cleanup identity")
	}
	if !filepath.IsAbs(rootPath) || filepath.Clean(rootPath) == filepath.Dir(filepath.Clean(rootPath)) {
		return errors.New("invalid managed image output root")
	}
	if err := imageWriteAllowed(rootPath); err != nil {
		return err
	}
	for path := filepath.Clean(rootPath); ; path = filepath.Dir(path) {
		if info, err := os.Lstat(path); err == nil && info.Mode()&os.ModeSymlink != 0 {
			return errors.New("managed image output root contains a symlink or reparse point")
		} else if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if filepath.Dir(path) == path {
			break
		}
	}
	root, err := os.OpenRoot(rootPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer root.Close()
	for parent := filepath.Dir(relative); parent != "."; parent = filepath.Dir(parent) {
		info, statErr := root.Lstat(parent)
		if errors.Is(statErr, os.ErrNotExist) {
			return nil
		}
		if statErr != nil {
			return statErr
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return errors.New("owned image path contains a symlink or reparse point")
		}
	}
	if recursive {
		err = root.RemoveAll(relative)
	} else {
		err = root.Remove(relative)
	}
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

func (v *imageService) deleteImageSessionFilesLocked(id string, legacy []string) (int, int, int, error) {
	if v.ownedDirectories == nil {
		v.ownedDirectories = map[string]imageOwnedDirectory{}
	}
	owners := map[string]string{}
	for _, job := range v.jobs {
		owners[job.ID] = job.SessionID
	}
	for key, asset := range v.assets {
		if asset.SessionID == "" && owners[asset.JobID] != "" {
			asset.SessionID = owners[asset.JobID]
			v.assets[key] = asset
		}
	}
	for _, key := range legacy {
		asset, exists := v.assets[key]
		if exists && asset.SessionID == "" && owners[asset.JobID] == "" {
			asset.SessionID = id
			v.assets[key] = asset
		}
	}
	protected := map[string]bool{}
	for key, asset := range v.assets {
		if asset.SessionID == "" || !v.deletedSessions[asset.SessionID] {
			protected[key] = true
		}
	}
	for _, job := range v.jobs {
		if !v.deletedSessions[job.SessionID] {
			for _, ref := range imageJobReferences(job) {
				protected[ref] = true
			}
		}
	}
	// Preserve cross-chat ancestry even after bounded job history was pruned.
	// A later deletion collects it when its final referencing chat disappears.
	pending := make([]string, 0, len(protected))
	for key := range protected {
		pending = append(pending, key)
	}
	for len(pending) > 0 {
		key := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		asset := v.assets[key]
		for _, ref := range append(append([]string{}, asset.ReferenceAssetIDs...), asset.ParentAssetID) {
			if ref != "" && !protected[ref] {
				protected[ref] = true
				pending = append(pending, ref)
			}
		}
	}
	type directory struct{ root, relative string }
	directories := map[directory]bool{}
	retained, removed := 0, 0
	for key, asset := range v.assets {
		if asset.SessionID == "" || !v.deletedSessions[asset.SessionID] {
			continue
		}
		if protected[key] {
			retained++
			continue
		}
		validName := asset.Filename == "image.png" || !asset.Input && imageBatchOutputName(asset.Filename) || asset.Input && (asset.Filename == "image.jpg" || asset.Filename == "image.webp")
		if !imageSafeID(asset.JobID) || !validName || key != asset.JobID+"/"+asset.Filename || filepath.Clean(asset.Path) != filepath.Clean(filepath.Join(asset.OutputRoot, asset.relativePath())) {
			return 0, removed, retained, errors.New("invalid managed image asset identity during deletion")
		}
		if err := removeImageOwnedPath(asset.OutputRoot, asset.relativePath(), false); err != nil {
			return 0, removed, retained, fmt.Errorf("cannot remove owned image asset: %w", err)
		}
		directories[directory{asset.OutputRoot, filepath.Dir(asset.relativePath())}] = true
		delete(v.assets, key)
		removed++
	}
	for _, job := range v.jobs {
		if job.SessionID == id && job.Kind == "generate" && imageSafeID(job.ID) {
			root := job.Config.OutputDir
			if root == "" {
				root = v.config.OutputDir
			}
			directories[directory{root, job.ID}] = true
		}
	}
	for jobID, dir := range v.ownedDirectories {
		if v.deletedSessions[dir.SessionID] {
			directories[directory{dir.Root, jobID}] = true
		}
	}
	for dir := range directories {
		shared := false
		for _, asset := range v.assets {
			if filepath.Clean(asset.OutputRoot) == filepath.Clean(dir.root) && filepath.Dir(asset.relativePath()) == dir.relative {
				shared = true
				break
			}
		}
		if !shared {
			if err := removeImageOwnedPath(dir.root, dir.relative, true); err != nil {
				return 0, removed, retained, fmt.Errorf("cannot remove owned image directory: %w", err)
			}
			if owned, exists := v.ownedDirectories[dir.relative]; exists && filepath.Clean(owned.Root) == filepath.Clean(dir.root) {
				delete(v.ownedDirectories, dir.relative)
			}
		}
	}
	if err := writeImageJSON(filepath.Join(v.root, "session-directories.json"), v.ownedDirectories); err != nil {
		return 0, removed, retained, err
	}
	if err := writeImageJSON(filepath.Join(v.root, "assets.json"), v.assets); err != nil {
		return 0, removed, retained, err
	}
	kept := make([]*imageJob, 0, len(v.jobs))
	removedJobs := 0
	for _, job := range v.jobs {
		if job.SessionID == id {
			removedJobs++
		} else {
			kept = append(kept, job)
		}
	}
	if err := writeImageJSON(filepath.Join(v.root, "jobs.json"), kept); err != nil {
		return 0, removed, retained, err
	}
	v.jobs = kept
	return removedJobs, removed, retained, nil
}
