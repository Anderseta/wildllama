package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
)

// Video settings are independent of the active image and chat profiles.
type videoProfile struct {
	Diffusion  string  `json:"diffusion_model"`
	Encoder    string  `json:"text_encoder"`
	VAE        string  `json:"vae"`
	Vision     string  `json:"vision_encoder"`
	Vocabulary string  `json:"tokenizer,omitempty"`
	Width      int     `json:"width"`
	Height     int     `json:"height"`
	Steps      int     `json:"steps"`
	CFG        float64 `json:"cfg_scale"`
	imagegen.VideoOptions
	Reserve int `json:"vram_reserve_mib"`
	RAM     int `json:"max_ram_mib"`
}

func defaultVideoProfile() videoProfile {
	return videoProfile{Width: 832, Height: 480, Steps: 4, CFG: 1, VideoOptions: imagegen.VideoOptions{Frames: 33, FPS: 16}, Reserve: imagegen.DefaultVRAMReserveMiB}
}

func (p videoProfile) runtimeProfile(mode string) imageProfile {
	model := imagegen.WanRapidI2V
	if mode == "t2v" {
		model = imagegen.WanRapidT2V
	}
	return imageProfile{Name: "Wan Rapid v10", ExecutionMode: "exclusive", Profile: imagegen.Profile{ID: model, ModelID: model,
		DiffusionModel: p.Diffusion, TextEncoder: p.Encoder, VAE: p.VAE, VisionEncoder: p.Vision, Tokenizer: p.Vocabulary,
		Width: p.Width, Height: p.Height, Steps: p.Steps, CFGScale: p.CFG, Sampler: "euler_a", Scheduler: "beta",
		OffloadToCPU: true, StageComponents: true, FlashAttention: true, VAETiling: true, MaxRAMMiB: p.RAM, VRAMReserveMiB: p.Reserve}}
}

func (p videoProfile) validate(mode string) error {
	if err := p.VideoOptions.Validate(); err != nil {
		return err
	}
	if err := imagegen.ValidateProfile(p.runtimeProfile(mode).Profile); err != nil {
		return err
	}
	for _, path := range []string{p.Diffusion, p.Encoder, p.VAE, p.Vision} {
		if path == "" {
			continue
		}
		ext := strings.ToLower(filepath.Ext(path))
		if !filepath.IsAbs(path) || (ext != ".gguf" && ext != ".safetensors") || strings.ContainsAny(path, "\x00\r\n") {
			return errors.New("select an absolute GGUF or safetensors component path")
		}
	}
	return nil
}

type videoRecord struct {
	ID      string    `json:"id"`
	Path    string    `json:"path"`
	Seed    int64     `json:"seed,string"`
	Created time.Time `json:"created_at"`
	Frames  int       `json:"frames"`
	FPS     int       `json:"fps"`
}

func (v *imageService) registerVideoResult(job *imageJob, result imagegen.Result) error {
	r := videoRecord{ID: job.ID, Path: result.Path, Seed: result.Seed, Created: time.Now().UTC(), Frames: job.Video.Frames, FPS: job.Video.FPS}
	if err := writeImageJSON(filepath.Join(v.root, "video-records", job.ID+".json"), r); err != nil {
		return err
	}
	v.mu.Lock()
	job.VideoFile = &r
	v.mu.Unlock()
	return nil
}

func (v *imageService) loadVideoProfile(mode string) (videoProfile, error) {
	p := defaultVideoProfile()
	data, err := os.ReadFile(filepath.Join(v.root, "video-"+mode+".json"))
	if errors.Is(err, os.ErrNotExist) {
		return p, nil
	}
	if err == nil {
		err = json.Unmarshal(data, &p)
	}
	return p, err
}
func videoMode(c *gin.Context) (string, bool) {
	mode := c.DefaultQuery("mode", "i2v")
	if mode != "i2v" && mode != "t2v" {
		c.JSON(400, gin.H{"error": "unknown video mode"})
		return "", false
	}
	return mode, true
}

func (s *Server) registerVideoRoutes(r *gin.Engine) {
	r.GET("/api/videos/config", s.videoConfigHandler)
	r.PUT("/api/videos/config", s.videoConfigHandler)
	r.POST("/api/videos/install", s.videoInstallHandler)
	r.GET("/api/videos/jobs", s.videoJobsHandler)
	r.POST("/api/videos/jobs", s.videoStartHandler)
	r.POST("/api/videos/jobs/:id/cancel", s.videoCancelHandler)
	r.GET("/api/videos/files/:id", s.videoFileHandler)
	r.HEAD("/api/videos/files/:id", s.videoFileHandler)
}
func (s *Server) videoConfigHandler(c *gin.Context) {
	mode, ok := videoMode(c)
	if !ok {
		return
	}
	v := s.imageService()
	v.videoProfileMu.Lock()
	defer v.videoProfileMu.Unlock()
	if c.Request.Method == http.MethodPut {
		var p videoProfile
		if !imageBind(c, &p) {
			return
		}
		if err := p.validate(mode); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		if err := writeImageJSON(filepath.Join(v.root, "video-"+mode+".json"), p); err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
	}
	profile, err := v.loadVideoProfile(mode)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	v.mu.Lock()
	out := v.config.OutputDir
	v.mu.Unlock()
	c.JSON(200, gin.H{"profile": profile, "output_dir": filepath.Join(out, "videos")})
}
func (s *Server) videoJobsHandler(c *gin.Context) {
	v := s.imageService()
	jobs := make([]*imageJob, 0)
	v.mu.Lock()
	for i := len(v.jobs) - 1; i >= 0; i-- {
		if job := v.jobs[i]; isVideoJob(job) {
			copy := cloneImageJobLocked(job)
			copy.Generation = nil
			jobs = append(jobs, copy)
		}
	}
	v.mu.Unlock()
	c.JSON(200, gin.H{"jobs": jobs})
}
func (s *Server) videoStartHandler(c *gin.Context) {
	mode, ok := videoMode(c)
	if !ok {
		return
	}
	var req struct {
		RequestID string `json:"request_id"`
		Prompt    string `json:"prompt"`
		Negative  string `json:"negative_prompt"`
		Asset     string `json:"input_asset_id"`
		Seed      *int64 `json:"seed"`
	}
	if !imageBind(c, &req) {
		return
	}
	if !imageSafeID(req.RequestID) {
		c.JSON(400, gin.H{"error": "missing or invalid request_id"})
		return
	}
	v := s.imageService()
	fingerprint := imageFingerprint([]any{"video", mode, req})
	v.mu.Lock()
	config := cloneImageConfig(v.config)
	asset, exists := v.assets[req.Asset]
	for _, job := range v.jobs {
		if job.RequestID == req.RequestID {
			copy := cloneImageJobLocked(job)
			v.mu.Unlock()
			if copy.Fingerprint != fingerprint {
				c.JSON(409, gin.H{"error": "request_id already used"})
				return
			}
			c.JSON(202, gin.H{"job": copy})
			return
		}
	}
	v.mu.Unlock()
	p, err := v.loadVideoProfile(mode)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	if err = p.validate(mode); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	profile := p.runtimeProfile(mode)
	request := imageJobRequest{RequestID: req.RequestID, Prompt: req.Prompt, NegativePrompt: req.Negative, Seed: req.Seed, InputAssetID: req.Asset}
	probe := imagegen.Request{Video: &p.VideoOptions, Prompt: req.Prompt, NegativePrompt: req.Negative}
	if req.Asset != "" {
		if !exists {
			c.JSON(400, gin.H{"error": "reference image not found"})
			return
		}
		handle, err := openImageAsset(asset)
		if err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		handle.Close()
		probe.ReferenceImages = []string{filepath.Join(asset.OutputRoot, asset.relativePath())}
	}
	if err = imagegen.ValidateRequest(profile.Profile, probe); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	generation, err := imageGenerationFor(request, profile)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	config.OutputDir = filepath.Join(config.OutputDir, "videos")
	job, err := v.enqueue(&imageJob{Kind: "video", RequestID: req.RequestID, ProfileID: profile.ID, ModelID: profile.ModelID,
		Request: request, Config: config, Profile: profile, Fingerprint: fingerprint, Generation: generation, Video: &p.VideoOptions, ParentAssetID: req.Asset})
	if err != nil {
		c.JSON(409, gin.H{"error": err.Error()})
		return
	}
	v.writeJob(c, 202, job)
}
func (s *Server) videoCancelHandler(c *gin.Context) {
	v := s.imageService()
	v.mu.Lock()
	found := false
	for _, job := range v.jobs {
		if job.ID == c.Param("id") && isVideoJob(job) {
			found = true
			break
		}
	}
	v.mu.Unlock()
	if !found {
		c.JSON(404, gin.H{"error": "video job not found"})
		return
	}
	s.imageCancelHandler(c)
}

func (s *Server) videoFileHandler(c *gin.Context) {
	id := c.Param("id")
	if !imageSafeID(id) {
		c.Status(404)
		return
	}
	v := s.imageService()
	data, err := os.ReadFile(filepath.Join(v.root, "video-records", id+".json"))
	var record videoRecord
	if err != nil || json.Unmarshal(data, &record) != nil || record.ID != id || filepath.Base(record.Path) != "video.webm" || filepath.Base(filepath.Dir(record.Path)) != id {
		c.Status(404)
		return
	}
	dir, err := os.OpenRoot(filepath.Dir(filepath.Dir(record.Path)))
	if err != nil {
		c.Status(404)
		return
	}
	defer dir.Close()
	file, err := dir.Open(filepath.Join(id, "video.webm"))
	if err != nil {
		c.Status(404)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		c.Status(404)
		return
	}
	c.Header("Content-Type", "video/webm")
	c.Header("X-Content-Type-Options", "nosniff")
	http.ServeContent(c.Writer, c.Request, fmt.Sprintf("%s_%d.webm", id, record.Seed), info.ModTime(), file)
}
