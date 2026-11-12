package server

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/imagegen"
)

type imageProfile struct {
	imagegen.Profile
	Name           string `json:"name"`
	PromptPrefix   string `json:"prompt_prefix,omitempty"`
	NegativePrompt string `json:"negative_prompt,omitempty"`
	ExecutionMode  string `json:"execution_mode,omitempty"`
	Seed           *int64 `json:"seed,omitempty"`
}

// Profile has its own missing-field defaults. Decode the outer fields
// explicitly so the embedded decoder cannot swallow names, prompts or seeds.
func (p *imageProfile) UnmarshalJSON(data []byte) error {
	var base imagegen.Profile
	if err := json.Unmarshal(data, &base); err != nil {
		return err
	}
	var extra struct {
		Name           string `json:"name"`
		PromptPrefix   string `json:"prompt_prefix"`
		NegativePrompt string `json:"negative_prompt"`
		ExecutionMode  string `json:"execution_mode"`
		Seed           *int64 `json:"seed"`
	}
	if err := json.Unmarshal(data, &extra); err != nil {
		return err
	}
	*p = imageProfile{Profile: base, Name: extra.Name, PromptPrefix: extra.PromptPrefix,
		NegativePrompt: extra.NegativePrompt, ExecutionMode: extra.ExecutionMode, Seed: extra.Seed}
	return nil
}

type imageConfig struct {
	Version           int    `json:"version"`
	Revision          uint64 `json:"revision"`
	DefaultProfileID  string `json:"default_profile_id"`
	ExecutionMode     string `json:"execution_mode"`
	RestoreTextModels bool   `json:"restore_text_models"`
	ReserveVRAMMiB    int    `json:"reserve_vram_mib"`
	imagegen.Config
	Profiles []imageProfile `json:"profiles"`
}

type imageJobRequest struct {
	RequestID        string   `json:"request_id,omitempty"`
	SessionID        string   `json:"session_id,omitempty"`
	ProfileID        string   `json:"profile_id,omitempty"`
	Prompt           string   `json:"prompt"`
	NegativePrompt   string   `json:"negative_prompt,omitempty"`
	Width            int      `json:"width,omitempty"`
	Height           int      `json:"height,omitempty"`
	Seed             *int64   `json:"seed"`
	Steps            int      `json:"steps,omitempty"`
	CFGScale         *float64 `json:"cfg_scale,omitempty"`
	Sampler          string   `json:"sampler,omitempty"`
	ExecutionMode    string   `json:"execution_mode,omitempty"`
	ReferenceImages  []string `json:"reference_images,omitempty"`
	InputAssetID     string   `json:"input_asset_id,omitempty"`
	Count            int      `json:"count,omitempty"`
	DeferTextRestore bool     `json:"defer_text_restore,omitempty"`
	ParentAssetID    string   `json:"parent_asset_id,omitempty"`
}

type imageProgress struct {
	Current    int64  `json:"current"`
	Total      int64  `json:"total"`
	Message    string `json:"message"`
	Role       string `json:"role,omitempty"`
	Filename   string `json:"filename,omitempty"`
	FileIndex  int    `json:"file_index,omitempty"`
	FileCount  int    `json:"file_count,omitempty"`
	BatchIndex int    `json:"batch_index,omitempty"`
	BatchCount int    `json:"batch_count,omitempty"`
}

type imageAsset struct {
	AssetID         string  `json:"asset_id"`
	RequestID       string  `json:"request_id,omitempty"`
	SessionID       string  `json:"session_id,omitempty"`
	JobID           string  `json:"job_id"`
	Filename        string  `json:"filename"`
	MIMEType        string  `json:"mime_type"`
	Path            string  `json:"path"`
	Width           int     `json:"width"`
	Height          int     `json:"height"`
	Seed            int64   `json:"seed"`
	DurationSeconds float64 `json:"duration_seconds"`
	BatchIndex      int     `json:"batch_index,omitempty"`
	BatchCount      int     `json:"batch_count,omitempty"`
	ParentAssetID   string  `json:"parent_asset_id,omitempty"`
}

type imageJob struct {
	ID                  string                  `json:"id"`
	Kind                string                  `json:"kind"`
	RequestID           string                  `json:"request_id,omitempty"`
	SessionID           string                  `json:"session_id,omitempty"`
	ProfileID           string                  `json:"profile_id,omitempty"`
	ModelID             string                  `json:"model_id,omitempty"`
	Status              string                  `json:"status"`
	Phase               string                  `json:"phase"`
	EffectiveMode       string                  `json:"effective_mode,omitempty"`
	ComputeBackend      string                  `json:"compute_backend,omitempty"`
	RAMOffload          bool                    `json:"ram_offload,omitempty"`
	Timings             *imageJobTimings        `json:"timings,omitempty"`
	Memory              *imageJobMemory         `json:"memory,omitempty"`
	Progress            imageProgress           `json:"progress"`
	CreatedAt           time.Time               `json:"created_at"`
	StartedAt           *time.Time              `json:"started_at,omitempty"`
	FinishedAt          *time.Time              `json:"finished_at,omitempty"`
	Error               string                  `json:"error,omitempty"`
	PersistenceError    string                  `json:"persistence_error,omitempty"`
	Result              *imageAsset             `json:"result,omitempty"`
	Results             []imageAsset            `json:"results,omitempty"`
	DeletedAssetIDs     []string                `json:"deleted_asset_ids,omitempty"`
	Generation          *imageGenerationSummary `json:"generation,omitempty"`
	ParentAssetID       string                  `json:"parent_asset_id,omitempty"`
	TextRestoreDeferred bool                    `json:"text_restore_deferred,omitempty"`
	RestoreErrors       []string                `json:"restore_errors,omitempty"`
	Fingerprint         string                  `json:"fingerprint,omitempty"`
	Request             imageJobRequest         `json:"-"`
	Config              imageConfig             `json:"-"`
	Profile             imageProfile            `json:"-"`
	RecipeID            string                  `json:"-"`
	IncludeVision       bool                    `json:"-"`
	Copy                *imageCopyRequest       `json:"-"`
	LoRA                *imagegen.Component     `json:"-"`
	cancel              context.CancelFunc
	ctx                 context.Context
	executing           bool
	timingPhase         string
	timingPhaseStarted  time.Time
}

type imageService struct {
	mu                sync.Mutex
	libraryMu         sync.Mutex
	server            *Server
	root              string
	config            imageConfig
	configError       string
	jobs              []*imageJob
	assets            map[string]imageAssetRecord
	gate              *imageGPUGate
	queue             chan *imageJob
	queueChanged      chan struct{}
	transferQueue     chan *imageJob
	transferChanged   chan struct{}
	transferStarted   bool
	workers           sync.WaitGroup
	active            imagegen.Runner
	closed            bool
	workerStarted     bool
	ctx               context.Context
	shutdownCancel    context.CancelFunc
	cancelledRequests map[string]time.Time
	uploadSlot        chan struct{}
	deletedSessions   map[string]bool
	deletedRequests   map[string]bool
	sessionDeleteErr  error
	assetOwnersDirty  bool
	assetCatalogError error
	assetDeletions    map[string]imageAssetRecord
	assetDeleteErr    error
	ownedDirectories  map[string]imageOwnedDirectory
	newEngine         func(imagegen.Config) (imagegen.Runner, error)
	text              imageTextBackend
}

func imageRoot() string {
	if path := strings.TrimSpace(os.Getenv("WILDLLAMA_IMAGES_DIR")); path != "" {
		return filepath.Clean(path)
	}
	return filepath.Join(filepath.Dir(envconfig.Models()), "images")
}

func defaultImageConfig(root string) imageConfig {
	executable := "sd-cli"
	if runtime.GOOS == "windows" {
		executable += ".exe"
	}
	profile := imagegen.DefaultProfile(imagegen.DefaultRecipeID)
	return imageConfig{Version: 1, Revision: 1, DefaultProfileID: profile.ID, ExecutionMode: "exclusive", RestoreTextModels: true,
		ReserveVRAMMiB: imagegen.DefaultVRAMReserveMiB, Config: imagegen.Config{ConditioningCacheMiB: 256, MinFreeDiskMiB: 1024, RunnerPath: filepath.Join(filepath.Dir(root), "lib", "wildimage", executable),
			WorkDir: filepath.Join(root, "work"), ModelsDir: filepath.Join(root, "models"), OutputDir: filepath.Join(root, "outputs")},
		Profiles: []imageProfile{{Profile: profile, Name: "Qwen Image 2.1 Turbo"}}}
}

func (s *Server) imageService() *imageService {
	s.imagesOnce.Do(func() {
		service := &imageService{server: s, root: imageRoot(), gate: newImageGPUGate(), queue: make(chan *imageJob, 16), assets: map[string]imageAssetRecord{}, cancelledRequests: map[string]time.Time{}, uploadSlot: make(chan struct{}, 1)}
		service.ctx, service.shutdownCancel = context.WithCancel(context.Background())
		service.config = defaultImageConfig(service.root)
		service.newEngine = func(config imagegen.Config) (imagegen.Runner, error) { return imagegen.NewEngine(config) }
		service.text = &schedulerImageText{server: s}
		if s.sched != nil {
			s.sched.imageGate.Store(service.gate)
		}
		if data, err := os.ReadFile(filepath.Join(service.root, "config.json")); err == nil {
			if err = json.Unmarshal(data, &service.config); err == nil {
				err = validateImageConfig(service.config)
			}
			if err != nil {
				service.configError = "Invalid saved image configuration: " + err.Error()
				service.config = defaultImageConfig(service.root)
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			service.configError = err.Error()
		}
		if err := service.recoverImageLibraryDeletionLocked(); err != nil {
			service.configError = "Image library cleanup requires attention: " + err.Error()
		}
		if data, err := os.ReadFile(filepath.Join(service.root, "jobs.json")); err == nil {
			if json.Unmarshal(data, &service.jobs) != nil {
				service.configError = "Invalid saved image job history"
				service.jobs = nil
			}
			valid := service.jobs[:0]
			for _, job := range service.jobs {
				if job == nil || !imageSafeID(job.ID) {
					continue
				}
				valid = append(valid, job)
				if len(job.Results) == 0 && job.Result != nil {
					job.Results = []imageAsset{*job.Result}
				}
				if job.Result == nil && len(job.Results) > 0 {
					first := job.Results[0]
					job.Result = &first
				}
				if !imageTerminal(job.Status) {
					now := time.Now().UTC()
					job.Status = "interrupted"
					job.Phase = "interrupted"
					job.Error = "WildLlama restarted before this job completed; it was not restarted automatically."
					job.FinishedAt = &now
				}
			}
			service.jobs = valid
		}
		if data, err := os.ReadFile(filepath.Join(service.root, "assets.json")); err == nil {
			if json.Unmarshal(data, &service.assets) != nil {
				service.configError = "Invalid saved image asset catalog"
				service.assetCatalogError = errors.New(service.configError)
				service.assets = map[string]imageAssetRecord{}
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			service.assetCatalogError = err
		}
		if service.assets == nil {
			service.assets = map[string]imageAssetRecord{}
		}
		if err := service.loadImageCatalog(); err != nil {
			service.assetCatalogError = err
			service.configError = "Cannot load image asset catalog: " + err.Error()
		}
		service.recoverImageJobStates()
		service.loadDeletedImageSessions()
		service.loadImageAssetDeletions()
		s.images = service
	})
	return s.images
}

func imageTerminal(status string) bool {
	return status == "completed" || status == "failed" || status == "cancelled" || status == "interrupted"
}

func cloneImageConfig(config imageConfig) imageConfig {
	config.Profiles = append([]imageProfile(nil), config.Profiles...)
	for i := range config.Profiles {
		config.Profiles[i] = cloneImageProfile(config.Profiles[i])
	}
	return config
}

func cloneImageProfile(profile imageProfile) imageProfile {
	if profile.Seed != nil {
		value := *profile.Seed
		profile.Seed = &value
	}
	profile.LoRAs = append([]imagegen.LoRA(nil), profile.LoRAs...)
	return profile
}

func imageSafeID(id string) bool {
	if len(id) == 0 || len(id) > 128 {
		return false
	}
	for _, char := range id {
		if !(char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z' || char >= '0' && char <= '9' || char == '-' || char == '_') {
			return false
		}
	}
	return true
}

// Profile identifiers are metadata, never filesystem paths. Preserve decimal
// version names such as the engine's qwen-image-2.1-turbo default.
func imageSafeProfileID(id string) bool {
	return id != "." && id != ".." && imageSafeID(strings.ReplaceAll(id, ".", "-"))
}

func newImageID() string {
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return fmt.Sprintf("image-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(id[:])
}

func validateImageConfig(config imageConfig) error {
	if config.Version != 1 || len(config.Profiles) > 128 || config.ReserveVRAMMiB < 0 || config.ReserveVRAMMiB > 1024*1024 || config.ConditioningCacheMiB < 0 || config.ConditioningCacheMiB > 8192 || config.MinFreeDiskMiB < 0 || config.MinFreeDiskMiB > 1024*1024 {
		return errors.New("invalid image configuration bounds")
	}
	if !validImageMode(config.ExecutionMode, false) {
		return errors.New("invalid image execution mode")
	}
	for _, path := range []string{config.RunnerPath, config.WorkDir, config.ModelsDir, config.OutputDir} {
		if !filepath.IsAbs(path) {
			return errors.New("image paths must be absolute")
		}
	}
	seen := map[string]bool{}
	for _, profile := range config.Profiles {
		if !imageSafeProfileID(profile.ID) || seen[profile.ID] || len(profile.Name) > 256 || len(profile.PromptPrefix) > 32768 || len(profile.NegativePrompt) > 32768 || !validImageMode(profile.ExecutionMode, true) {
			return errors.New("invalid or duplicate image profile")
		}
		if profile.Seed != nil && *profile.Seed < -1 {
			return errors.New("image profile seed must be -1, null or nonnegative")
		}
		if err := imagegen.ValidateProfile(profile.Profile); err != nil {
			return fmt.Errorf("profile %s: %w", profile.ID, err)
		}
		seen[profile.ID] = true
	}
	if config.DefaultProfileID != "" && !seen[config.DefaultProfileID] {
		return errors.New("default image profile does not exist")
	}
	return nil
}

func validImageMode(mode string, empty bool) bool {
	return empty && mode == "" || mode == "exclusive" || mode == "simultaneous" || mode == "auto"
}

func imageWriteAllowed(path string) error {
	if runtime.GOOS == "windows" {
		resolved := filepath.Clean(path)
		for ancestor := resolved; ; ancestor = filepath.Dir(ancestor) {
			if target, err := filepath.EvalSymlinks(ancestor); err == nil {
				resolved = target
				break
			}
			if filepath.Dir(ancestor) == ancestor {
				break
			}
		}
		if strings.EqualFold(filepath.VolumeName(resolved), "C:") {
			return errors.New("image configuration, assets and caches must be stored outside C:")
		}
	}
	return nil
}

func writeImageJSON(path string, value any) error {
	if err := imageWriteAllowed(path); err != nil {
		return err
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".image-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err = tmp.Chmod(0600); err == nil {
		_, err = tmp.Write(data)
	}
	if err == nil {
		err = tmp.Sync()
	}
	closeErr := tmp.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(name, path)
}

func imageBind(c *gin.Context, value any) bool {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 1<<20)
	decoder := json.NewDecoder(c.Request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		c.AbortWithStatusJSON(400, gin.H{"error": err.Error()})
		return false
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		c.AbortWithStatusJSON(400, gin.H{"error": "expected one JSON object"})
		return false
	}
	return true
}

func (v *imageService) runtime() imagegen.RunnerStatus {
	v.mu.Lock()
	engine := v.active
	config := v.config.Config
	savedError := v.configError
	v.mu.Unlock()
	if engine != nil {
		return engine.Status()
	}
	if savedError != "" {
		return imagegen.RunnerStatus{RunnerPath: config.RunnerPath, Error: savedError}
	}
	engine, err := v.newEngine(config)
	if err != nil {
		return imagegen.RunnerStatus{RunnerPath: config.RunnerPath, Error: err.Error()}
	}
	defer engine.Close()
	return engine.Status()
}

func (s *Server) registerImageRoutes(r *gin.Engine) {
	r.GET("/api/images/config", s.imageConfigHandler)
	r.PATCH("/api/images/config", s.imageConfigHandler)
	r.GET("/api/images/models", s.imageModelsHandler)
	r.POST("/api/images/models/import", s.imageImportHandler)
	r.POST("/api/images/models/inventory", s.imageInventoryHandler)
	r.POST("/api/images/models/copy", s.imageCopyHandler)
	r.GET("/api/images/library", s.imageLibraryHandler)
	r.POST("/api/images/library/delete", s.imageLibraryDeleteHandler)
	r.GET("/api/images/loras", s.imageLoRAsHandler)
	r.GET("/api/images/loras/files", s.imageLoRAFilesHandler)
	r.POST("/api/images/loras/install", s.imageLoRAInstallHandler)
	r.POST("/api/images/models/install", s.imageInstallHandler)
	r.GET("/api/images/jobs", s.imageJobsHandler)
	r.POST("/api/images/jobs", s.imageJobsHandler)
	r.GET("/api/images/jobs/:id", s.imageJobHandler)
	r.POST("/api/images/jobs/:id/cancel", s.imageCancelHandler)
	r.POST("/api/images/jobs/by-request/:request_id/cancel", s.imageCancelHandler)
	r.GET("/api/images/assets/:id/:filename", s.imageAssetHandler)
	r.DELETE("/api/images/assets/:id/:filename", s.imageAssetDeleteHandler)
	r.POST("/api/images/assets", s.imageUploadHandler)
	r.DELETE("/api/images/sessions/:session_id", s.imageSessionDeleteHandler)
	r.GET("/api/images/capabilities", func(c *gin.Context) {
		c.JSON(200, gin.H{"session_deletion": true, "asset_deletion": true, "model_library": true})
	})
}

func (s *Server) imageConfigHandler(c *gin.Context) {
	v := s.imageService()
	if c.Request.Method == http.MethodPatch {
		v.libraryMu.Lock()
		defer v.libraryMu.Unlock()
		var patch map[string]json.RawMessage
		if !imageBind(c, &patch) {
			return
		}
		v.mu.Lock()
		for _, job := range v.jobs {
			if !imageTerminal(job.Status) {
				v.mu.Unlock()
				c.JSON(409, gin.H{"error": "image jobs must finish or be cancelled before changing configuration"})
				return
			}
		}
		data, _ := json.Marshal(v.config)
		var values map[string]json.RawMessage
		_ = json.Unmarshal(data, &values)
		if revision, ok := patch["revision"]; ok {
			var expected uint64
			if json.Unmarshal(revision, &expected) != nil || expected != v.config.Revision {
				v.mu.Unlock()
				c.JSON(409, gin.H{"error": "image configuration changed; refresh before saving"})
				return
			}
		}
		for key, value := range patch {
			if _, ok := values[key]; !ok {
				v.mu.Unlock()
				c.JSON(400, gin.H{"error": "unknown image configuration field: " + key})
				return
			}
			values[key] = value
		}
		data, _ = json.Marshal(values)
		var next imageConfig
		err := json.Unmarshal(data, &next)
		if err == nil {
			next.Version = 1
			next.Revision = v.config.Revision + 1
			err = validateImageConfig(next)
		}
		if err == nil {
			err = writeImageJSON(filepath.Join(v.root, "config.json"), next)
		}
		if err != nil {
			v.mu.Unlock()
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		v.config = next
		v.configError = ""
		v.mu.Unlock()
	}
	v.mu.Lock()
	config := cloneImageConfig(v.config)
	v.mu.Unlock()
	c.JSON(200, gin.H{"config": config, "config_path": filepath.Join(v.root, "config.json"), "runtime": v.runtime()})
}

func (s *Server) imageModelsHandler(c *gin.Context) {
	v := s.imageService()
	v.mu.Lock()
	root := v.config.ModelsDir
	v.mu.Unlock()
	models, recipes, err := imagegen.Catalog(root)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	if models == nil {
		models = []imagegen.Bundle{}
	}
	c.JSON(200, gin.H{"models": models, "recipes": recipes, "runtime": v.runtime()})
}

func (s *Server) imageImportHandler(c *gin.Context) {
	var request struct {
		ManifestPath string `json:"manifest_path"`
	}
	if !imageBind(c, &request) {
		return
	}
	v := s.imageService()
	v.libraryMu.Lock()
	defer v.libraryMu.Unlock()
	v.mu.Lock()
	config := cloneImageConfig(v.config)
	v.mu.Unlock()
	bundle, err := imagegen.ImportLocal(c.Request.Context(), config.ModelsDir, request.ManifestPath)
	if err == nil {
		err = v.recordBundle(bundle)
	}
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"model": bundle})
}

func imageFingerprint(value any) string {
	data, _ := json.Marshal(value)
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}
