package server

import (
	"errors"
	"path/filepath"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
)

func isVideoJob(job *imageJob) bool { return job.Kind == "video" || job.Kind == "video_install" }

func (s *Server) videoInstallHandler(c *gin.Context) {
	var req struct {
		RequestID string `json:"request_id"`
	}
	if !imageBind(c, &req) {
		return
	}
	if !imageSafeID(req.RequestID) {
		c.JSON(400, gin.H{"error": "missing or invalid request_id"})
		return
	}
	v := s.imageService()
	p, err := v.loadVideoProfile("i2v")
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	v.mu.Lock()
	cfg := cloneImageConfig(v.config)
	v.mu.Unlock()
	job, err := v.enqueue(&imageJob{Kind: "video_install", ModelID: imagegen.WanI2VPackageID, RequestID: req.RequestID, Fingerprint: imageFingerprint([]string{imagegen.WanI2VPackageID, req.RequestID}), Config: cfg, Profile: p.runtimeProfile("i2v")})
	if err != nil {
		c.JSON(409, gin.H{"error": err.Error()})
		return
	}
	v.writeJob(c, 202, job)
}

func (v *imageService) installWanVideo(job *imageJob) error {
	p := job.Profile.Profile
	paths, err := imagegen.InstallWanI2V(job.ctx, job.Config.ModelsDir, map[string]string{
		"diffusion": p.DiffusionModel, "text_encoder": p.TextEncoder, "vae": p.VAE, "vision_encoder": p.VisionEncoder, "wan_vocabulary": p.Tokenizer,
	}, func(progress imagegen.DownloadProgress) { v.updateDownload(job, progress) })
	if err != nil {
		return err
	}
	return v.applyWanVideoPackage(job, paths)
}

func (v *imageService) applyWanVideoPackage(job *imageJob, paths map[string]string) error {
	p := job.Profile.Profile
	v.videoProfileMu.Lock()
	defer v.videoProfileMu.Unlock()
	if err := job.ctx.Err(); err != nil {
		return err
	}
	latest, err := v.loadVideoProfile("i2v")
	if err != nil {
		return err
	}
	if latest.Diffusion != p.DiffusionModel || latest.Encoder != p.TextEncoder || latest.VAE != p.VAE || latest.Vision != p.VisionEncoder || latest.Vocabulary != p.Tokenizer {
		return errors.New("Pacote baixado. Os componentes do perfil foram editados durante o download. Clique em baixar novamente para aplicar os arquivos verificados.")
	}
	latest.Diffusion = paths["diffusion"]
	latest.Encoder = paths["text_encoder"]
	latest.VAE = paths["vae"]
	latest.Vision = paths["vision_encoder"]
	latest.Vocabulary = paths["wan_vocabulary"]
	return writeImageJSON(filepath.Join(v.root, "video-i2v.json"), latest)
}
