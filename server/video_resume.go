package server

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
)

func (j *imageJob) videoCheckpointPath() string {
	if j.Video == nil || j.Generation == nil || j.Generation.Profile.ModelID != imagegen.WanRapidI2V || !imageSafeID(j.ID) || !filepath.IsAbs(j.VideoWorkDir) {
		return ""
	}
	return filepath.Join(j.VideoWorkDir, "video-checkpoints", j.ID+".wanstep")
}

func (j *imageJob) latentSize() int64 {
	p := j.Generation.Profile
	return imagegen.WanLatentBytes(p.Width, p.Height, j.Video.Frames)
}

func (j *imageJob) resumableStep() int {
	if j.executing || !imageTerminal(j.Status) || j.HistoryCleared || j.VideoFile != nil {
		return 0
	}
	return max(j.diskStep, j.checkpoint.Step())
}

func (j *imageJob) discoverSavedStep() {
	path := j.videoCheckpointPath()
	if path == "" {
		return
	}
	if j.HistoryCleared || j.VideoFile != nil || j.Status == "completed" {
		// Finish an explicit discard or publication interrupted just after the
		// history was committed. This job ID uniquely owns its file.
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			j.PersistenceError = err.Error()
		}
		return
	}
	n, err := imagegen.SavedVideoStep(path, j.latentSize(), j.Generation.Profile.Steps)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		j.PersistenceError = err.Error()
	}
	j.diskStep = n
}

func (j *imageJob) dropVideoCheckpoint() error {
	j.checkpoint.Close()
	j.checkpoint = nil
	if j.diskStep > 0 {
		if err := os.Remove(j.videoCheckpointPath()); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		j.diskStep = 0
	}
	return nil
}

// Only the GPU worker loads saved steps, so queued or duplicate requests do not
// allocate checkpoint RAM or hold up unrelated media operations.
func (v *imageService) prepareVideoCheckpoint(job *imageJob) (*imagegen.VideoCheckpoint, error) {
	v.mu.Lock()
	if err := job.ctx.Err(); err != nil {
		v.mu.Unlock()
		return nil, err
	}
	if !job.isResume || job.diskStep <= job.checkpoint.Step() {
		if job.checkpoint == nil {
			job.checkpoint = &imagegen.VideoCheckpoint{}
		}
		checkpoint := job.checkpoint
		v.mu.Unlock()
		return checkpoint, nil
	}
	path, size, steps := job.videoCheckpointPath(), job.latentSize(), job.Generation.Profile.Steps
	v.mu.Unlock()

	loaded, loadErr := imagegen.LoadVideoCheckpoint(path, size, steps)
	v.mu.Lock()
	if err := job.ctx.Err(); err != nil {
		v.mu.Unlock()
		loaded.Close()
		return nil, err
	}
	if loadErr != nil {
		v.mu.Unlock()
		return nil, fmt.Errorf("não foi possível carregar o passo salvo para retomar o vídeo: %w", loadErr)
	}
	previous := job.checkpoint
	job.checkpoint = loaded
	v.mu.Unlock()
	previous.Close()
	return loaded, nil
}

// Called exclusively on shutdown, after all workers have joined. A checkpoint
// from a previous normal close remains usable throughout a subsequent attempt.
func (v *imageService) saveVideoCheckpoints() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	var failures error
	for _, job := range v.jobs {
		if job.checkpoint.Step() == 0 || job.executing || job.HistoryCleared || job.VideoFile != nil {
			continue
		}
		n := job.checkpoint.Step()
		if n > job.diskStep {
			if err := job.checkpoint.Save(job.videoCheckpointPath()); err != nil {
				failures = errors.Join(failures, err)
				continue
			}
			job.diskStep = n
		}
		job.checkpoint.Close()
		job.checkpoint = nil
	}
	return failures
}

func (s *Server) videoShutdownHandler(c *gin.Context) {
	host, _, err := net.SplitHostPort(c.Request.RemoteAddr)
	if err != nil || !net.ParseIP(host).IsLoopback() {
		c.JSON(http.StatusForbidden, gin.H{"error": "somente encerramento local"})
		return
	}
	if err := s.shutdownMedia(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"saved": true})
}

func (s *Server) videoResumeHandler(c *gin.Context) {
	v := s.imageService()
	v.mu.Lock()
	defer v.mu.Unlock()
	for i, job := range v.jobs {
		if job.ID != c.Param("id") || job.Video == nil {
			continue
		}
		if v.closed {
			c.JSON(409, gin.H{"error": "O Wildllama está encerrando."})
			return
		}
		if !imageTerminal(job.Status) && job.isResume {
			c.JSON(202, gin.H{"job": cloneImageJobLocked(job)})
			return
		}
		if job.resumableStep() == 0 {
			c.JSON(409, gin.H{"error": "Nenhum passo completo disponível para retomar."})
			return
		}
		v.compactImageQueueLocked()
		if len(v.queue) == cap(v.queue) {
			c.JSON(409, gin.H{"error": errImageGPUQueueFull.Error()})
			return
		}
		if _, ok := v.assets[job.ParentAssetID]; !ok {
			c.JSON(409, gin.H{"error": "A imagem de referência original está indisponível."})
			return
		}
		next := *job
		next.Config = cloneImageConfig(v.config)
		next.Config.WorkDir = job.VideoWorkDir
		next.Config.OutputDir = filepath.Join(next.Config.OutputDir, "videos")
		next.Profile = job.Generation.Profile
		next.Request = imageJobRequest{RequestID: job.RequestID, Prompt: job.Generation.UserPrompt, NegativePrompt: job.Generation.UserNegativePrompt, InputAssetID: job.ParentAssetID}
		next.Status, next.Phase = "queued", "queued"
		next.Error, next.PersistenceError, next.VideoMemoryIssue = "", "", ""
		next.RestoreErrors = nil
		if job.Timings != nil {
			next.PriorVideoSeconds += job.Timings.RunSeconds
		}
		queued := time.Now().UTC()
		next.queuedAgain = &queued
		next.StartedAt, next.FinishedAt, next.Timings = nil, nil, nil
		next.Progress = imageProgress{}
		next.isResume = true
		next.ctx, next.cancel = context.WithCancel(v.ctx)
		next.ctx = imagegen.WithDiskReserve(next.ctx, next.Config.MinFreeDiskMiB)
		v.jobs[i] = &next
		if err := v.persistJobsLocked(); err != nil {
			next.cancel()
			v.jobs[i] = job
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		if !v.workerStarted {
			v.workerStarted = true
			v.workers.Add(1)
			go func() { defer v.workers.Done(); v.work() }()
		}
		v.queue <- &next
		v.notifyImageQueueLocked()
		c.JSON(202, gin.H{"job": cloneImageJobLocked(&next)})
		return
	}
	c.JSON(404, gin.H{"error": "Vídeo não encontrado."})
}
