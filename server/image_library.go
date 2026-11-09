package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
)

// Library identities are server-generated path hashes. Destructive requests never
// accept a client-supplied path, and never recursively remove a model directory.
type imageLibraryComponent struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	Path         string   `json:"path"`
	Roles        []string `json:"roles"`
	Size         int64    `json:"size"`
	Ready        bool     `json:"ready"`
	Managed      bool     `json:"managed"`
	SourceURL    string   `json:"source_url,omitempty"`
	ProfileIDs   []string `json:"profile_ids"`
	ProfileNames []string `json:"profile_names"`
	PipelineIDs  []string `json:"pipeline_ids"`
	CanDelete    bool     `json:"can_delete"`
	DeleteReason string   `json:"delete_reason,omitempty"`
}

type imageLibraryPipeline struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	ProfileIDs   []string `json:"profile_ids"`
	ProfileNames []string `json:"profile_names"`
	ComponentIDs []string `json:"component_ids"`
	CanRemove    bool     `json:"can_remove"`
	RemoveReason string   `json:"remove_reason,omitempty"`
	directory    string
}

type imageLibrary struct {
	ModelsDir  string                   `json:"models_dir"`
	Revision   uint64                   `json:"revision"`
	Components []*imageLibraryComponent `json:"components"`
	Pipelines  []*imageLibraryPipeline  `json:"pipelines"`
	Truncated  bool                     `json:"truncated"`
}

func imageLibraryPathKey(path string) string {
	path = filepath.Clean(path)
	if runtime.GOOS == "windows" {
		path = strings.ToLower(path)
	}
	return path
}

func imageLibraryID(path string) string {
	hash := sha256.Sum256([]byte(imageLibraryReferenceKey(path)))
	return hex.EncodeToString(hash[:])
}

func imageLibraryReferenceKey(path string) string {
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		return imageLibraryPathKey(resolved)
	}
	return imageLibraryPathKey(path)
}

func appendImageUnique(values []string, value string) []string {
	if value == "" {
		return values
	}
	for _, existing := range values {
		if value == existing {
			return values
		}
	}
	return append(values, value)
}

func imageProfileComponents(profile imagegen.Profile, visit func(string, string)) {
	visit(profile.DiffusionModel, "diffusion_model")
	visit(profile.TextEncoder, "text_encoder")
	visit(profile.VAE, "vae")
	visit(profile.VisionEncoder, "vision_encoder")
	for _, lora := range profile.LoRAs {
		visit(lora.Path, "lora")
	}
}

// Require every ancestor and the final entry to be a plain path. OpenRoot below
// also confines the actual rename/unlink if another process races this check.
func imageLibrarySafePath(root, path string) error {
	root, path = filepath.Clean(root), filepath.Clean(path)
	if !filepath.IsAbs(root) || root == filepath.Dir(root) || !filepath.IsAbs(path) {
		return errors.New("invalid managed image library root")
	}
	relative, err := filepath.Rel(root, path)
	if err != nil || relative == "." || !filepath.IsLocal(relative) {
		return errors.New("external files are preserved")
	}
	if err := imageWriteAllowed(path); err != nil {
		return err
	}
	for ancestor := path; ; ancestor = filepath.Dir(ancestor) {
		info, err := os.Lstat(ancestor)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if err == nil && info.Mode()&(os.ModeSymlink|os.ModeIrregular) != 0 {
			return errors.New("linked or reparse paths cannot be deleted")
		}
		if filepath.Dir(ancestor) == ancestor {
			break
		}
	}
	return nil
}

func imageLibrarySource(component imagegen.Component) string {
	if validateImageLoRASource(component.Repo, component.Revision, "", false) != nil || component.Revision == "" || validateCreateFilePath(component.Filename) != nil || strings.Contains(component.Filename, "\\") {
		return ""
	}
	parts := strings.Split(component.Filename, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return "https://huggingface.co/" + component.Repo + "/blob/" + component.Revision + "/" + strings.Join(parts, "/")
}

// Caller holds mu, stabilizing configuration and job admission during the small
// metadata-only scan. No model hashes, tensor payloads, network or GPU are read.
func (v *imageService) imageLibraryLocked() (imageLibrary, error) {
	config := v.config
	result := imageLibrary{ModelsDir: config.ModelsDir, Revision: config.Revision,
		Components: []*imageLibraryComponent{}, Pipelines: []*imageLibraryPipeline{}}
	if err := v.recoverImageLibraryDeletionLocked(); err != nil {
		return result, err
	}
	busy := v.closed
	for _, job := range v.jobs {
		busy = busy || job.executing || !imageTerminal(job.Status)
	}
	components := map[string]*imageLibraryComponent{}
	add := func(path, role string) *imageLibraryComponent {
		if path == "" || !filepath.IsAbs(path) {
			return nil
		}
		key := imageLibraryReferenceKey(path)
		component := components[key]
		if component == nil {
			component = &imageLibraryComponent{ID: imageLibraryID(path), Name: filepath.Base(path), Path: filepath.Clean(path),
				Roles: []string{}, ProfileIDs: []string{}, ProfileNames: []string{}, PipelineIDs: []string{}}
			component.Managed = imageLibrarySafePath(config.ModelsDir, path) == nil
			if info, err := os.Lstat(path); err == nil && info.Mode().IsRegular() {
				component.Size, component.Ready = info.Size(), info.Size() > 0
			}
			components[key] = component
			result.Components = append(result.Components, component)
		}
		component.Roles = appendImageUnique(component.Roles, role)
		return component
	}
	models, _, err := imagegen.Catalog(config.ModelsDir)
	if err != nil {
		return result, err
	}
	if len(models) >= 512 {
		result.Truncated = true
	}
	pipelineCounts := map[string]int{}
	for _, model := range models {
		pipelineCounts[model.ID]++
	}
	for _, model := range models {
		if len(model.Components) == 0 {
			// An unreadable manifest may still reference another pipeline's files.
			// Fail closed for physical deletion rather than guessing ownership.
			result.Truncated = true
		}
		pipeline := &imageLibraryPipeline{ID: model.ID, Name: model.Name, ProfileIDs: []string{}, ProfileNames: []string{}, ComponentIDs: []string{}, directory: model.Directory}
		if pipeline.Name == "" {
			pipeline.Name = model.ID
		}
		pipeline.CanRemove = !busy && imageLibrarySafePath(config.ModelsDir, filepath.Join(model.Directory, imagegen.ManifestName)) == nil
		if pipelineCounts[model.ID] > 1 || filepath.Base(model.Directory) != model.ID {
			pipeline.CanRemove = false
			result.Truncated = true
		}
		if !pipeline.CanRemove {
			pipeline.RemoveReason = "Finish or cancel image jobs before removing a managed pipeline."
		}
		for _, component := range model.Components {
			entry := add(component.Path, component.Role)
			if entry == nil {
				continue
			}
			entry.SourceURL = imageLibrarySource(component.Component)
			entry.Ready = entry.Ready && component.Ready
			entry.PipelineIDs = appendImageUnique(entry.PipelineIDs, model.ID)
			pipeline.ComponentIDs = appendImageUnique(pipeline.ComponentIDs, entry.ID)
		}
		imageProfileComponents(model.Profile, func(path, role string) {
			if entry := add(path, role); entry != nil {
				entry.PipelineIDs = appendImageUnique(entry.PipelineIDs, model.ID)
				pipeline.ComponentIDs = appendImageUnique(pipeline.ComponentIDs, entry.ID)
			}
		})
		for _, profile := range config.Profiles {
			if profile.ModelID == model.ID {
				pipeline.ProfileIDs = append(pipeline.ProfileIDs, profile.ID)
				pipeline.ProfileNames = append(pipeline.ProfileNames, profile.Name)
			}
		}
		result.Pipelines = append(result.Pipelines, pipeline)
	}
	for _, profile := range config.Profiles {
		imageProfileComponents(profile.Profile, func(path, role string) {
			if entry := add(path, role); entry != nil {
				entry.ProfileIDs = appendImageUnique(entry.ProfileIDs, profile.ID)
				entry.ProfileNames = appendImageUnique(entry.ProfileNames, profile.Name)
			}
		})
	}
	// Include unused components as well as registered pipelines. Bounded traversal
	// skips incoming downloads, links, and opaque directories; it never recurses
	// outside the configured model root.
	visited := 0
	err = filepath.WalkDir(config.ModelsDir, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			if errors.Is(walkErr, os.ErrNotExist) && path == config.ModelsDir {
				return nil
			}
			return walkErr
		}
		visited++
		if visited > 16384 {
			result.Truncated = true
			return fs.SkipAll
		}
		if entry.Type()&(os.ModeSymlink|os.ModeIrregular) != 0 {
			if entry.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if entry.IsDir() {
			if path != config.ModelsDir && strings.HasPrefix(entry.Name(), ".") {
				return fs.SkipDir
			}
			// Guard the root and junction ancestors before WalkDir opens them.
			if path == config.ModelsDir {
				if imageLibrarySafePath(config.ModelsDir, filepath.Join(path, "library-probe")) != nil {
					return fs.SkipDir
				}
			} else if imageLibrarySafePath(config.ModelsDir, path) != nil {
				return fs.SkipDir
			}
			return nil
		}
		extension := strings.ToLower(filepath.Ext(path))
		if entry.Name() == "wildimage-lora.json" {
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			var asset imagegen.LoRAAsset
			err = json.NewDecoder(io.LimitReader(file, 65536)).Decode(&asset)
			file.Close()
			if err == nil && filepath.Clean(asset.Path) == filepath.Join(filepath.Dir(path), "adapter"+strings.ToLower(filepath.Ext(asset.Filename))) {
				if component := add(asset.Path, "lora"); component != nil {
					component.SourceURL = imageLibrarySource(imagegen.Component{Repo: asset.Repo, Revision: asset.Revision, Filename: asset.Filename})
				}
			}
		}
		if extension == ".gguf" || extension == ".safetensors" {
			role := localImageRole(path)
			if role == "" {
				if existing := components[imageLibraryReferenceKey(path)]; existing != nil && len(existing.Roles) > 0 {
					return nil
				}
				role = "unknown"
			}
			add(path, role)
		}
		return nil
	})
	if err != nil {
		return result, err
	}
	for _, component := range result.Components {
		if len(component.Roles) > 1 {
			filtered := component.Roles[:0]
			for _, role := range component.Roles {
				if role != "unknown" {
					filtered = append(filtered, role)
				}
			}
			component.Roles = filtered
		}
		sort.Strings(component.Roles)
		switch {
		case !component.Managed:
			component.DeleteReason = "External files and linked paths are preserved."
		case busy:
			component.DeleteReason = "Finish or cancel image jobs before deleting components."
		case len(component.ProfileIDs) > 0 || len(component.PipelineIDs) > 0:
			component.DeleteReason = "Referenced by a pipeline or profile; remove its references first."
		case result.Truncated:
			component.DeleteReason = "The library scan was incomplete; deletion is disabled."
		default:
			component.CanDelete = true
		}
	}
	sort.Slice(result.Components, func(i, j int) bool { return result.Components[i].Path < result.Components[j].Path })
	return result, nil
}

func (s *Server) imageLibraryHandler(c *gin.Context) {
	v := s.imageService()
	v.libraryMu.Lock()
	defer v.libraryMu.Unlock()
	v.mu.Lock()
	library, err := v.imageLibraryLocked()
	v.mu.Unlock()
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, library)
}

func (s *Server) imageLibraryDeleteHandler(c *gin.Context) {
	var request struct {
		Kind            string `json:"kind"`
		ID              string `json:"id"`
		DeleteFiles     bool   `json:"delete_files"`
		Revision        uint64 `json:"revision"`
		ConfirmProfiles bool   `json:"confirm_profiles"`
	}
	if !imageBind(c, &request) {
		return
	}
	if request.Kind != "component" && request.Kind != "pipeline" || request.ID == "" {
		c.JSON(400, gin.H{"error": "select an image library component or pipeline"})
		return
	}
	v := s.imageService()
	v.libraryMu.Lock()
	defer v.libraryMu.Unlock()
	v.mu.Lock()
	defer v.mu.Unlock()
	if request.Revision != v.config.Revision {
		c.JSON(409, gin.H{"error": "image configuration changed; refresh before deleting"})
		return
	}
	library, err := v.imageLibraryLocked()
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	next := cloneImageConfig(v.config)
	paths := []string{}
	deletedFiles, retainedFiles := 0, 0
	if request.Kind == "component" {
		var selected *imageLibraryComponent
		for _, component := range library.Components {
			if component.ID == request.ID {
				selected = component
			}
		}
		if selected == nil {
			c.JSON(404, gin.H{"error": "image component no longer exists"})
			return
		}
		if !request.DeleteFiles || !selected.CanDelete {
			c.JSON(409, gin.H{"error": "only unreferenced managed component files can be deleted", "reason": selected.DeleteReason})
			return
		}
		paths = append(paths, selected.Path)
		deletedFiles = 1
		// Remove only the metadata belonging to a managed LoRA identity.
		if filepath.Base(filepath.Dir(filepath.Dir(selected.Path))) == "loras" {
			paths = append(paths, filepath.Join(filepath.Dir(selected.Path), "wildimage-lora.json"))
		}
	} else {
		var selected *imageLibraryPipeline
		for _, pipeline := range library.Pipelines {
			if pipeline.ID == request.ID {
				selected = pipeline
			}
		}
		if selected == nil {
			c.JSON(404, gin.H{"error": "image pipeline no longer exists"})
			return
		}
		if !selected.CanRemove || library.Truncated && request.DeleteFiles {
			c.JSON(409, gin.H{"error": "finish image jobs and refresh the complete library before removing a pipeline"})
			return
		}
		if len(selected.ProfileIDs) > 0 && !request.ConfirmProfiles {
			c.JSON(409, gin.H{"error": "confirm removal of this pipeline's image profiles", "profile_ids": selected.ProfileIDs})
			return
		}
		keptProfiles := make([]imageProfile, 0, len(next.Profiles))
		removedProfiles := map[string]bool{}
		for _, profile := range next.Profiles {
			if profile.ModelID == selected.ID {
				removedProfiles[profile.ID] = true
			} else {
				keptProfiles = append(keptProfiles, profile)
			}
		}
		next.Profiles = keptProfiles
		if removedProfiles[next.DefaultProfileID] {
			next.DefaultProfileID = ""
			if len(keptProfiles) > 0 {
				next.DefaultProfileID = keptProfiles[0].ID
			}
		}
		paths = append(paths, filepath.Join(selected.directory, imagegen.ManifestName))
		if request.DeleteFiles {
			paths = append(paths, filepath.Join(selected.directory, "local-copy-verification.json"))
		}
		for _, component := range library.Components {
			belongs, shared := false, false
			for _, pipeline := range component.PipelineIDs {
				belongs = belongs || pipeline == selected.ID
				shared = shared || pipeline != selected.ID
			}
			if !belongs {
				continue
			}
			for _, profile := range component.ProfileIDs {
				shared = shared || !removedProfiles[profile]
			}
			if request.DeleteFiles && component.Managed && !shared {
				paths = append(paths, component.Path)
				deletedFiles++
			} else {
				retainedFiles++
			}
		}
	}
	next.Revision++
	if err := validateImageConfig(next); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	if err := v.removeImageLibraryPathsLocked(paths, next); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	updated, err := v.imageLibraryLocked()
	if err != nil {
		c.JSON(200, gin.H{"removed": true, "deleted_files": deletedFiles, "retained_files": retainedFiles, "config": next, "refresh_error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"removed": true, "deleted_files": deletedFiles, "retained_files": retainedFiles, "config": next, "library": updated})
}

type imageLibraryMovedFile struct {
	Original  string `json:"original"`
	Temporary string `json:"temporary"`
}

type imageLibraryDeletion struct {
	ModelsDir string                  `json:"models_dir"`
	Revision  uint64                  `json:"revision"`
	Moved     []imageLibraryMovedFile `json:"moved"`
}

// A small durable receipt makes a crash between rename, config publication and
// unlink recoverable: restore before publication, finish deletion afterwards.
func (v *imageService) recoverImageLibraryDeletionLocked() error {
	receiptPath := filepath.Join(v.root, "library-delete.json")
	file, err := os.Open(receiptPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	var receipt imageLibraryDeletion
	err = json.NewDecoder(io.LimitReader(file, 4<<20)).Decode(&receipt)
	file.Close()
	if err != nil || receipt.ModelsDir != v.config.ModelsDir || len(receipt.Moved) > 16384 || receipt.Revision == 0 {
		return errors.New("invalid pending image library cleanup receipt")
	}
	for _, moved := range receipt.Moved {
		if !filepath.IsLocal(moved.Original) || !filepath.IsLocal(moved.Temporary) || filepath.Dir(moved.Original) != filepath.Dir(moved.Temporary) || !strings.HasPrefix(filepath.Base(moved.Temporary), ".delete-") {
			return errors.New("invalid pending library cleanup identity")
		}
		for _, path := range []string{moved.Original, moved.Temporary} {
			if err := imageLibrarySafePath(receipt.ModelsDir, filepath.Join(receipt.ModelsDir, path)); err != nil {
				return err
			}
		}
	}
	root, err := os.OpenRoot(receipt.ModelsDir)
	if err != nil {
		return err
	}
	defer root.Close()
	for _, moved := range receipt.Moved {
		info, err := root.Lstat(moved.Temporary)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil || !info.Mode().IsRegular() {
			return errors.New("pending library cleanup entry changed")
		}
		if v.config.Revision >= receipt.Revision {
			err = root.Remove(moved.Temporary)
		} else {
			if _, err := root.Lstat(moved.Original); !errors.Is(err, os.ErrNotExist) {
				return errors.New("library cleanup cannot restore an occupied path")
			}
			err = root.Rename(moved.Temporary, moved.Original)
		}
		if err != nil {
			return err
		}
	}
	return os.Remove(receiptPath)
}

func (v *imageService) removeImageLibraryPathsLocked(paths []string, next imageConfig) error {
	rootPath := v.config.ModelsDir
	for _, path := range paths {
		if err := imageLibrarySafePath(rootPath, path); err != nil {
			return err
		}
	}
	root, err := os.OpenRoot(rootPath)
	if err != nil {
		return err
	}
	defer root.Close()
	receipt := imageLibraryDeletion{ModelsDir: rootPath, Revision: next.Revision, Moved: []imageLibraryMovedFile{}}
	for _, path := range paths {
		relative, _ := filepath.Rel(rootPath, path)
		info, err := root.Lstat(relative)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil || !info.Mode().IsRegular() {
			return fmt.Errorf("library entry changed before deletion: %s", relative)
		}
		temporary := filepath.Join(filepath.Dir(relative), ".delete-"+newImageID())
		receipt.Moved = append(receipt.Moved, imageLibraryMovedFile{relative, temporary})
	}
	receiptPath := filepath.Join(v.root, "library-delete.json")
	if err := writeImageJSON(receiptPath, receipt); err != nil {
		return err
	}
	rollback := func() {
		_ = v.recoverImageLibraryDeletionLocked()
	}
	for _, moved := range receipt.Moved {
		if err := root.Rename(moved.Original, moved.Temporary); err != nil {
			rollback()
			return err
		}
	}
	if err := writeImageJSON(filepath.Join(v.root, "config.json"), next); err != nil {
		rollback()
		return err
	}
	v.config = next
	if err := v.recoverImageLibraryDeletionLocked(); err != nil {
		return err
	}
	for _, file := range receipt.Moved {
		// Only empty directories are removed. No original, unrecognized file or
		// shared component can disappear as a side effect of pipeline cleanup.
		if directory := filepath.Dir(file.Original); directory != "." {
			_ = root.Remove(directory)
		}
	}
	return nil
}
