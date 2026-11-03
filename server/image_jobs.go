package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
)

type imageAssetRecord struct {
	imageAsset
	OutputRoot        string `json:"output_root"`
	Input             bool   `json:"input,omitempty"`
	UploadFingerprint string `json:"upload_fingerprint,omitempty"`
}

func (v *imageService) persistJobsLocked() error {
	// Bound history metadata without deleting output files or their asset index.
	jobs := v.jobs
	if len(v.jobs) > 200 {
		remove := len(v.jobs) - 200
		kept := make([]*imageJob, 0, len(v.jobs))
		for _, job := range v.jobs {
			if remove > 0 && imageTerminal(job.Status) {
				remove--
				continue
			}
			kept = append(kept, job)
		}
		jobs = kept
	}
	if err := writeImageJSON(filepath.Join(v.root, "jobs.json"), jobs); err != nil {
		return err
	}
	v.jobs = jobs
	return nil
}

func (v *imageService) update(job *imageJob, phase, message string, current, total int64) {
	v.mu.Lock()
	defer v.mu.Unlock()
	job.Phase = phase
	job.Progress = imageProgress{Current: current, Total: total, Message: message, BatchIndex: job.Progress.BatchIndex, BatchCount: job.Progress.BatchCount}
	// Byte/step progress is live; persisted running/terminal states avoid disk churn.
}

func (v *imageService) enqueue(job *imageJob) (*imageJob, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.closed {
		return nil, errors.New("image service is shutting down")
	}
	if job.RequestID != "" {
		if !imageSafeID(job.RequestID) {
			return nil, errors.New("invalid image request_id")
		}
		if expires := v.cancelledRequests[job.RequestID]; expires.After(time.Now()) {
			return nil, errors.New("this image request was cancelled before submission completed")
		}
		for _, existing := range v.jobs {
			if existing.RequestID == job.RequestID {
				if existing.Fingerprint != job.Fingerprint {
					return nil, errors.New("request_id already belongs to a different image request")
				}
				return existing, nil
			}
		}
	}
	if len(v.queue) >= cap(v.queue) {
		v.compactImageQueueLocked()
		if len(v.queue) >= cap(v.queue) {
			return nil, errImageGPUQueueFull
		}
	}
	job.ID = newImageID()
	job.Status = "queued"
	job.Phase = "queued"
	job.CreatedAt = time.Now().UTC()
	job.ctx, job.cancel = context.WithCancel(v.ctx)
	previous := v.jobs
	v.jobs = append(v.jobs, job)
	if err := v.persistJobsLocked(); err != nil {
		v.jobs = previous
		job.cancel()
		return nil, err
	}
	if !v.workerStarted {
		v.workerStarted = true
		go v.work()
	}
	v.queue <- job
	v.notifyImageQueueLocked()
	return job, nil
}

func (v *imageService) notifyImageQueueLocked() {
	if v.queueChanged == nil {
		v.queueChanged = make(chan struct{}, 1)
	}
	select {
	case v.queueChanged <- struct{}{}:
	default:
	}
}

func (v *imageService) nextImageJob() (*imageJob, bool) {
	v.mu.Lock()
	for {
		select {
		case job, open := <-v.queue:
			v.mu.Unlock()
			return job, open
		default:
			changed := v.queueChanged
			v.mu.Unlock()
			<-changed
			v.mu.Lock()
		}
	}
}

func (v *imageService) compactImageQueueLocked() {
	if v.closed {
		return
	}
	// Dequeue uses the same mutex, so temporarily draining the channel cannot
	// let the worker take a later request ahead of an earlier survivor.
	count := len(v.queue)
	kept := make([]*imageJob, 0, count)
	for range count {
		job := <-v.queue
		if !imageTerminal(job.Status) {
			kept = append(kept, job)
		}
	}
	for _, job := range kept {
		v.queue <- job
	}
}

func (v *imageService) work() {
	for {
		job, open := v.nextImageJob()
		if !open {
			return
		}
		v.mu.Lock()
		if imageTerminal(job.Status) {
			v.mu.Unlock()
			continue
		}
		closed := v.closed
		if job.ctx.Err() != nil || closed {
			now := time.Now().UTC()
			job.Status = "cancelled"
			job.Phase = "cancelled"
			job.FinishedAt = &now
			_ = v.persistJobsLocked()
			v.mu.Unlock()
			continue
		}
		started := time.Now().UTC()
		job.StartedAt = &started
		job.Status = "running"
		_ = v.persistJobsLocked()
		v.mu.Unlock()
		var err error
		if job.Kind == "install_lora" {
			err = v.installLoRA(job)
		} else if job.Kind == "import" {
			err = v.copyLocal(job)
		} else if job.Kind == "install" {
			err = v.install(job)
		} else {
			err = v.generate(job)
		}
		v.mu.Lock()
		if job.TextRestoreDeferred && job.Status == "completed" {
			// Full successful delivery was committed while holding the lease.
			// A later cancellation cannot undo saved output or require a reload.
			job.cancel()
			v.mu.Unlock()
			continue
		}
		finished := time.Now().UTC()
		job.FinishedAt = &finished
		switch {
		case job.ctx.Err() != nil:
			job.Status = "cancelled"
			job.Phase = "cancelled"
		case err != nil:
			job.Status = "failed"
			job.Phase = "failed"
		default:
			job.Status = "completed"
			job.Phase = "completed"
		}
		if err != nil {
			job.Error = err.Error()
		}
		job.cancel()
		_ = v.persistJobsLocked()
		v.mu.Unlock()
	}
}

func (v *imageService) install(job *imageJob) error {
	bundle, err := imagegen.Install(job.ctx, job.Config.ModelsDir, job.RecipeID, job.IncludeVision, func(progress imagegen.DownloadProgress) {
		v.updateDownload(job, progress)
	})
	if err != nil {
		return err
	}
	return v.recordBundle(bundle)
}

func (v *imageService) updateDownload(job *imageJob, progress imagegen.DownloadProgress) {
	v.mu.Lock()
	defer v.mu.Unlock()
	job.Phase = progress.Phase
	job.Progress = imageProgress{Current: progress.Completed, Total: progress.Total, Message: progress.Message,
		Role: progress.Role, Filename: progress.Filename, FileIndex: progress.FileIndex, FileCount: progress.FileCount}
}

func (v *imageService) recordBundle(bundle imagegen.Bundle) error {
	v.mu.Lock()
	defer v.mu.Unlock()
	next := cloneImageConfig(v.config)
	matched := false
	for i := range next.Profiles {
		profile := &next.Profiles[i]
		if profile.ModelID != bundle.ID {
			continue
		}
		matched = true
		if profile.DiffusionModel == "" {
			profile.DiffusionModel = bundle.Profile.DiffusionModel
		}
		if profile.TextEncoder == "" {
			profile.TextEncoder = bundle.Profile.TextEncoder
		}
		if profile.VAE == "" {
			profile.VAE = bundle.Profile.VAE
		}
		if profile.VisionEncoder == "" {
			profile.VisionEncoder = bundle.Profile.VisionEncoder
		}
	}
	if !matched {
		profile := imageProfile{Profile: bundle.Profile, Name: bundle.Name}
		profile.LoRAs = append([]imagegen.LoRA(nil), bundle.Profile.LoRAs...)
		for _, existing := range next.Profiles {
			if existing.ID == profile.ID {
				profile.ID = "image-" + newImageID()
				break
			}
		}
		next.Profiles = append(next.Profiles, profile)
	}
	next.Revision++
	if err := writeImageJSON(filepath.Join(v.root, "config.json"), next); err != nil {
		return err
	}
	v.config = next
	return nil
}

func (v *imageService) generate(job *imageJob) (runErr error) {
	request := job.Request
	generation, err := imageGenerationFor(request, job.Profile)
	if err != nil {
		return err
	}
	if err := v.resolveImageGenerationDimensions(generation); err != nil {
		return err
	}
	profile := generation.Profile.Profile
	seed, err := imageBatchSeed(request, generation.Profile, generation.Count)
	if err != nil {
		return err
	}
	input := imagegen.Request{Prompt: generation.Prompt, NegativePrompt: generation.NegativePrompt, Width: profile.Width, Height: profile.Height, Seed: seed, OutputPath: filepath.Join(job.Config.OutputDir, job.ID, "image.png")}
	refs := generation.ReferenceAssetIDs
	parent := request.ParentAssetID
	if parent == "" && len(refs) > 0 {
		parent = refs[0]
	}
	if parent != "" {
		v.mu.Lock()
		_, exists := v.assets[parent]
		v.mu.Unlock()
		if !exists {
			return errors.New("parent image is not a registered image asset")
		}
	}
	for _, id := range refs {
		v.mu.Lock()
		asset, ok := v.assets[id]
		v.mu.Unlock()
		if !ok {
			return errors.New("reference image is not a registered image asset")
		}
		file, err := openImageAsset(asset)
		if err != nil {
			return err
		}
		file.Close()
		input.ReferenceImages = append(input.ReferenceImages, filepath.Join(asset.OutputRoot, asset.relativePath()))
	}
	if err := imagegen.ValidateRequest(profile, input); err != nil {
		return err
	}
	v.mu.Lock()
	job.Generation = generation
	job.ParentAssetID = parent
	job.Progress.BatchCount = generation.Count
	v.mu.Unlock()
	mode := request.ExecutionMode
	if mode == "" {
		mode = job.Profile.ExecutionMode
	}
	if mode == "" {
		mode = job.Config.ExecutionMode
	}
	if mode == "auto" {
		mode = "exclusive"
	}
	if !validImageMode(mode, false) {
		return errors.New("invalid image execution mode")
	}
	if mode == "simultaneous" && profile.MaxVRAMMiB <= 0 {
		return errors.New("simultaneous mode requires a positive image VRAM fitting target")
	}
	// Recheck the runtime after queueing, before disrupting resident text. This
	// performs no subprocess launch or tensor load, and closes on every path.
	engine, err := v.newEngine(job.Config.Config)
	if err != nil {
		return err
	}
	var release func()
	var saved []imageTextState
	var beforeImageFree uint64
	imageStarted, activeOwned := false, false
	// Cleanup owns its own deadline: cancelling image generation must still
	// release its process and restore unchanged prior text profiles when chosen.
	defer func() {
		if release != nil {
			defer release()
		}
		if closeErr := engine.Close(); closeErr != nil && runErr == nil {
			runErr = closeErr
		}
		needsRestore := false
		for _, state := range saved {
			if state.duration != 0 {
				needsRestore = true
				break
			}
		}
		v.mu.Lock()
		if activeOwned {
			v.active = nil
		}
		shuttingDown := v.closed
		restoreDeferred := false
		if job.Config.RestoreTextModels && needsRestore && !shuttingDown && request.DeferTextRestore && runErr == nil && job.ctx.Err() == nil && len(job.Results) == generation.Count {
			oldStatus, oldPhase, oldFinished := job.Status, job.Phase, job.FinishedAt
			finished := time.Now().UTC()
			job.Status, job.Phase = "completed", "completed"
			job.FinishedAt = &finished
			job.TextRestoreDeferred = true
			if err := v.persistJobsLocked(); err != nil {
				job.Status, job.Phase, job.FinishedAt = oldStatus, oldPhase, oldFinished
				job.TextRestoreDeferred = false
				runErr = errors.Join(runErr, fmt.Errorf("images saved but deferred completion could not be persisted: %w", err))
			} else {
				restoreDeferred = true
			}
		}
		v.mu.Unlock()
		if job.Config.RestoreTextModels && needsRestore && !shuttingDown && !restoreDeferred {
			v.update(job, "restoring_text", "Restoring previously resident text models and profiles", 0, 0)
			cleanup, cancel := context.WithTimeout(v.ctx, 10*time.Minute)
			if imageStarted && beforeImageFree > 0 {
				v.update(job, "restoring_text", "Waiting briefly for released image memory before restoring text", 0, 0)
				// Driver accounting can lag behind process exit. This is a short
				// best-effort wait, not a new quota: other programs may consume
				// memory, and the normal scheduler still gets to attempt restore.
				_ = v.waitForImageMemoryRecovery(cleanup, beforeImageFree)
			}
			v.update(job, "restoring_text", "Restoring previously resident text models and profiles", 0, 0)
			failures := v.text.Restore(cleanup, saved)
			cancel()
			v.mu.Lock()
			job.RestoreErrors = failures
			v.mu.Unlock()
		}
	}()
	if status := engine.Status(); !status.Ready {
		return fmt.Errorf("image runtime is not ready: %s", status.Error)
	}
	v.mu.Lock()
	job.EffectiveMode = mode
	v.mu.Unlock()
	v.update(job, "waiting_for_text", "Waiting for active text and RAG requests to finish", 0, 0)
	transition, stopWait := context.WithTimeout(job.ctx, 10*time.Minute)
	release, err = v.gate.exclusive(transition, job.ID)
	stopWait()
	if err != nil {
		return err
	}
	if mode == "exclusive" {
		v.update(job, "unloading_text", "Releasing text weights, context and draft memory", 0, 0)
		drain, cancel := context.WithTimeout(job.ctx, 2*time.Minute)
		saved, err = v.text.Suspend(drain)
		cancel()
		if err != nil {
			return err
		}
	}
	v.update(job, "checking_memory", "Checking actual free CUDA memory; the fitting target is not a hard process limit", 0, 0)
	if beforeImageFree, err = v.waitForMemory(job.ctx, profile.MaxVRAMMiB, job.Config.ReserveVRAMMiB); err != nil {
		return err
	}
	if mode == "simultaneous" {
		if err = v.gate.coexist(job.ID); err != nil {
			return err
		}
	}
	v.mu.Lock()
	v.active = engine
	activeOwned = true
	v.mu.Unlock()
	for index := 1; index <= generation.Count; index++ {
		if err := job.ctx.Err(); err != nil {
			return err
		}
		filename := imageBatchFilename(index)
		input.OutputPath = filepath.Join(job.Config.OutputDir, job.ID, filename)
		input.Seed = seed
		if seed >= 0 {
			input.Seed += int64(index - 1)
		}
		v.setImageBatchProgress(job, index, generation.Count)
		if index > 1 {
			v.update(job, "checking_memory", "Checking free CUDA memory for the next batch image", 0, 0)
			free, err := v.waitForMemory(job.ctx, profile.MaxVRAMMiB, job.Config.ReserveVRAMMiB)
			if err != nil {
				return err
			}
			beforeImageFree = free
		}
		v.update(job, "loading_image", "Preparing the isolated image runner", 0, 0)
		imageStarted = true
		result, generationErr := engine.Run(job.ctx, profile, input, func(progress imagegen.Progress) {
			v.update(job, progress.Phase, progress.Message, int64(progress.Current), int64(progress.Total))
		})
		if result.Path != "" {
			if err := v.registerImageResult(job, result, filename, index, generation.Count); err != nil {
				return errors.Join(generationErr, err)
			}
		} else if generationErr == nil {
			return errors.New("image runner returned no output")
		}
		if generationErr != nil {
			return generationErr
		}
	}
	return nil
}

func (v *imageService) waitForMemory(ctx context.Context, target, reserve int) (uint64, error) {
	needed := uint64(target+reserve) * 1024 * 1024
	check, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	for {
		free, err := v.text.FreeVRAM(check)
		if err != nil {
			return 0, err
		}
		if free >= needed {
			return free, nil
		}
		select {
		case <-check.Done():
			return 0, fmt.Errorf("insufficient free CUDA memory: fitting target plus headroom requires %d MiB: %w", target+reserve, check.Err())
		case <-time.After(500 * time.Millisecond):
		}
	}
}

func (v *imageService) waitForImageMemoryRecovery(ctx context.Context, before uint64) error {
	// Small display/driver fluctuations must not turn a normal restore into a
	// fixed three-second delay. Preserve the observed baseline, allowing 64 MiB.
	const tolerance = uint64(64 << 20)
	target := uint64(0)
	if before > tolerance {
		target = before - tolerance
	}
	check, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		free, err := v.text.FreeVRAM(check)
		if err != nil {
			return err
		}
		if free >= target {
			return nil
		}
		select {
		case <-check.Done():
			return check.Err()
		case <-ticker.C:
		}
	}
}

func (s *Server) imageInstallHandler(c *gin.Context) {
	var request struct {
		PipelineID    string `json:"pipeline_id"`
		RecipeID      string `json:"recipe_id,omitempty"`
		RequestID     string `json:"request_id,omitempty"`
		IncludeVision *bool  `json:"include_vision,omitempty"`
	}
	if !imageBind(c, &request) {
		return
	}
	if request.RecipeID != "" {
		if request.PipelineID != "" && request.PipelineID != request.RecipeID {
			c.JSON(400, gin.H{"error": "conflicting pipeline_id and recipe_id"})
			return
		}
		request.PipelineID = request.RecipeID
		request.RecipeID = ""
	}
	found := false
	for _, recipe := range imagegen.Recipes() {
		if recipe.ID == request.PipelineID {
			found = true
		}
	}
	if !found {
		c.JSON(400, gin.H{"error": "unknown image pipeline recipe"})
		return
	}
	v := s.imageService()
	v.mu.Lock()
	config := cloneImageConfig(v.config)
	v.mu.Unlock()
	includeVision := request.IncludeVision == nil || *request.IncludeVision
	job, err := v.enqueue(&imageJob{Kind: "install", ModelID: request.PipelineID, RequestID: request.RequestID, RecipeID: request.PipelineID, IncludeVision: includeVision, Config: config, Fingerprint: imageFingerprint(request)})
	if err != nil {
		c.JSON(409, gin.H{"error": err.Error()})
		return
	}
	v.writeJob(c, 202, job)
}

func (s *Server) imageJobsHandler(c *gin.Context) {
	v := s.imageService()
	if c.Request.Method == http.MethodGet {
		v.mu.Lock()
		defer v.mu.Unlock()
		jobs := make([]*imageJob, 0, len(v.jobs))
		requestID := c.Query("request_id")
		sessionID := c.Query("session_id")
		for i := len(v.jobs) - 1; i >= 0; i-- {
			if (requestID == "" || v.jobs[i].RequestID == requestID) && (sessionID == "" || v.jobs[i].SessionID == sessionID) {
				jobs = append(jobs, v.jobs[i])
			}
		}
		c.JSON(200, gin.H{"jobs": jobs})
		return
	}
	var request imageJobRequest
	if !imageBind(c, &request) {
		return
	}
	if strings.TrimSpace(request.Prompt) == "" || len(request.Prompt) > 32768 || len(request.NegativePrompt) > 32768 || len(request.SessionID) > 256 || len(request.ReferenceImages) > 4 || !validImageMode(request.ExecutionMode, true) || request.Seed != nil && *request.Seed < -1 || request.Count < 0 || request.Count > 8 || len(request.ParentAssetID) > 512 {
		c.JSON(400, gin.H{"error": "invalid image job request"})
		return
	}
	fingerprint := imageFingerprint(request)
	v.mu.Lock()
	config := cloneImageConfig(v.config)
	for _, job := range v.jobs {
		if request.RequestID != "" && job.RequestID == request.RequestID {
			if job.Fingerprint != fingerprint {
				v.mu.Unlock()
				c.JSON(409, gin.H{"error": "request_id belongs to a different image request"})
				return
			}
			c.JSON(202, gin.H{"job": job})
			v.mu.Unlock()
			return
		}
	}
	profileID := request.ProfileID
	if profileID == "" {
		profileID = config.DefaultProfileID
	}
	var profile imageProfile
	found := false
	for _, item := range config.Profiles {
		if item.ID == profileID {
			profile = item
			found = true
			break
		}
	}
	v.mu.Unlock()
	if !found {
		c.JSON(400, gin.H{"error": "image profile was not found"})
		return
	}
	if request.Steps != 0 {
		profile.Steps = request.Steps
	}
	if request.CFGScale != nil {
		profile.CFGScale = *request.CFGScale
	}
	if request.Sampler != "" {
		profile.Sampler = request.Sampler
	}
	if err := imagegen.ValidateProfile(profile.Profile); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	generation, err := imageGenerationFor(request, profile)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	if err := v.resolveImageGenerationDimensions(generation); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	parent := request.ParentAssetID
	if parent == "" && len(generation.ReferenceAssetIDs) > 0 {
		parent = generation.ReferenceAssetIDs[0]
	}
	if parent != "" {
		v.mu.Lock()
		_, exists := v.assets[parent]
		v.mu.Unlock()
		if !exists {
			c.JSON(400, gin.H{"error": "parent image is not a registered image asset"})
			return
		}
	}
	probe, err := v.newEngine(config.Config)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	status := probe.Status()
	_ = probe.Close()
	if !status.Ready {
		c.JSON(400, gin.H{"error": "image runtime is not ready: " + status.Error})
		return
	}
	job, err := v.enqueue(&imageJob{Kind: "generate", RequestID: request.RequestID, SessionID: request.SessionID, ProfileID: profile.ID, ModelID: profile.ModelID, Request: request, Config: config, Profile: profile, Fingerprint: fingerprint, Generation: generation, ParentAssetID: parent})
	if err != nil {
		c.JSON(409, gin.H{"error": err.Error()})
		return
	}
	v.writeJob(c, 202, job)
}

func (v *imageService) writeJob(c *gin.Context, status int, job *imageJob) {
	v.mu.Lock()
	defer v.mu.Unlock()
	c.JSON(status, gin.H{"job": job})
}

func (s *Server) imageJobHandler(c *gin.Context) {
	v := s.imageService()
	v.mu.Lock()
	defer v.mu.Unlock()
	for _, job := range v.jobs {
		if job.ID == c.Param("id") {
			c.JSON(200, gin.H{"job": job})
			return
		}
	}
	c.JSON(404, gin.H{"error": "image job not found"})
}

func (s *Server) imageCancelHandler(c *gin.Context) {
	v := s.imageService()
	v.mu.Lock()
	defer v.mu.Unlock()
	for _, job := range v.jobs {
		if job.ID != c.Param("id") && (c.Param("request_id") == "" || job.RequestID != c.Param("request_id")) {
			continue
		}
		if job.cancel != nil && !imageTerminal(job.Status) {
			job.cancel()
		}
		if job.Status == "queued" {
			now := time.Now().UTC()
			job.Status, job.Phase = "cancelled", "cancelled"
			job.FinishedAt = &now
			v.compactImageQueueLocked()
			if err := v.persistJobsLocked(); err != nil {
				c.JSON(500, gin.H{"error": "image request was cancelled but its status could not be persisted: " + err.Error(), "job": job})
				return
			}
		}
		c.JSON(200, gin.H{"job": job})
		return
	}
	if id := c.Param("request_id"); id != "" {
		if !imageSafeID(id) {
			c.JSON(400, gin.H{"error": "invalid image request_id"})
			return
		}
		now := time.Now()
		for key, expires := range v.cancelledRequests {
			if !expires.After(now) {
				delete(v.cancelledRequests, key)
			}
		}
		if len(v.cancelledRequests) >= 128 {
			oldest := ""
			var expires time.Time
			for key, value := range v.cancelledRequests {
				if oldest == "" || value.Before(expires) {
					oldest = key
					expires = value
				}
			}
			delete(v.cancelledRequests, oldest)
		}
		v.cancelledRequests[id] = now.Add(5 * time.Minute)
		c.JSON(200, gin.H{"cancelled": true, "request_id": id})
		return
	}
	c.JSON(404, gin.H{"error": "image job not found"})
}

func (s *Server) imageAssetHandler(c *gin.Context) {
	v := s.imageService()
	v.mu.Lock()
	asset, ok := v.assets[c.Param("id")+"/"+c.Param("filename")]
	v.mu.Unlock()
	if !ok {
		c.JSON(404, gin.H{"error": "image asset not found"})
		return
	}
	file, err := openImageAsset(asset)
	if err != nil {
		c.JSON(404, gin.H{"error": "image asset is unavailable"})
		return
	}
	defer file.Close()
	info, _ := file.Stat()
	c.Header("X-Content-Type-Options", "nosniff")
	c.Header("Cache-Control", "no-store")
	http.ServeContent(c.Writer, c.Request, asset.Filename, info.ModTime(), file)
}

func (s *Server) stopImageJobs() {
	v := s.imageService()
	v.mu.Lock()
	if v.closed {
		v.mu.Unlock()
		return
	}
	v.closed = true
	v.shutdownCancel()
	for _, job := range v.jobs {
		if job.cancel != nil && !imageTerminal(job.Status) {
			job.cancel()
		}
	}
	engine := v.active
	v.mu.Unlock()
	if engine != nil {
		_ = engine.Close()
	}
	v.gate.shutdown()
	v.mu.Lock()
	close(v.queue)
	v.notifyImageQueueLocked()
	v.mu.Unlock()
}
