package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
)

type imageLocalCandidate struct {
	Role            string `json:"role"`
	Path            string `json:"path"`
	Filename        string `json:"filename"`
	Bytes           int64  `json:"bytes"`
	Compatibility   string `json:"compatibility"`
	Hint            string `json:"hint"`
	Format          string `json:"format"`
	SupportsEditing bool   `json:"supports_editing,omitempty"`
}

type imageCopyComponent struct {
	Role            string   `json:"role"`
	Path            string   `json:"path"`
	Scale           *float64 `json:"scale,omitempty"`
	info            os.FileInfo
	supportsEditing bool
}

type imageCopyRequest struct {
	Name       string               `json:"name,omitempty"`
	RequestID  string               `json:"request_id,omitempty"`
	Components []imageCopyComponent `json:"components"`
	Profile    *imagegen.Profile    `json:"profile,omitempty"`
}

func localImageProfile() imagegen.Profile {
	profile := imagegen.DefaultProfile("")
	profile.ID = "qwen-image-2.1-local"
	profile.Steps, profile.CFGScale, profile.Sigmas = 25, 1, ""
	profile.Scheduler = "simple"
	profile.DiffusionModel, profile.TextEncoder, profile.VAE, profile.VisionEncoder = "", "", "", ""
	profile.StageComponents = true
	return profile
}

func localImageRole(path string) string {
	name := strings.ToLower(filepath.Base(path))
	folder := strings.ToLower(filepath.Base(filepath.Dir(path)))
	switch {
	case strings.Contains(name, "lora") || folder == "loras":
		return "lora"
	case strings.Contains(name, "mmproj") || folder == "clip_vision":
		return "vision_encoder"
	case strings.Contains(name, "vae") || folder == "vae":
		return "vae"
	case folder == "text_encoders" || folder == "clip" || strings.Contains(name, "qwen3vl"):
		return "text_encoder"
	case folder == "diffusion_models" || folder == "unet" || strings.Contains(name, "qwen-image"):
		return "diffusion_model"
	}
	return ""
}

func (s *Server) imageInventoryHandler(c *gin.Context) {
	var request struct {
		Directory string `json:"directory"`
	}
	if !imageBind(c, &request) {
		return
	}
	if !filepath.IsAbs(request.Directory) {
		c.JSON(400, gin.H{"error": "choose an absolute local model directory"})
		return
	}
	info, err := os.Stat(request.Directory)
	if err != nil || !info.IsDir() {
		c.JSON(400, gin.H{"error": "local model directory is unavailable"})
		return
	}
	root := filepath.Clean(request.Directory)
	candidates := make([]imageLocalCandidate, 0)
	visited := 0
	truncated := false
	var scan func(string, int) error
	scan = func(directory string, depth int) error {
		if err := c.Request.Context().Err(); err != nil {
			return err
		}
		file, err := os.Open(directory)
		if err != nil {
			return err
		}
		defer file.Close()
		for {
			entries, readErr := file.ReadDir(128)
			for _, entry := range entries {
				visited++
				if visited > 8192 || len(candidates) >= 512 {
					truncated = true
					return nil
				}
				path := filepath.Join(directory, entry.Name())
				if entry.Type()&os.ModeSymlink != 0 {
					continue
				}
				if entry.IsDir() {
					if depth < 3 && !strings.HasPrefix(entry.Name(), ".") {
						if err := scan(path, depth+1); err != nil {
							return err
						}
						if truncated {
							return nil
						}
					}
					continue
				}
				ext := strings.ToLower(filepath.Ext(path))
				if ext != ".gguf" && ext != ".safetensors" {
					continue
				}
				stat, err := entry.Info()
				if err != nil || !stat.Mode().IsRegular() {
					continue
				}
				role := localImageRole(path)
				candidate := imageLocalCandidate{Role: role, Path: path, Filename: entry.Name(), Bytes: stat.Size(), Compatibility: "unsupported", Hint: "Select the component role to validate compatibility"}
				if role != "" {
					inspection, err := imagegen.InspectLocalComponent(path, role)
					if err != nil {
						candidate.Hint = err.Error()
					} else {
						candidate.Compatibility, candidate.Hint, candidate.Format = inspection.Compatibility, inspection.Hint, inspection.Format
						candidate.SupportsEditing = inspection.SupportsEditing
					}
				}
				candidates = append(candidates, candidate)
			}
			if readErr == io.EOF {
				return nil
			}
			if readErr != nil {
				return readErr
			}
		}
	}
	if err = scan(root, 0); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	c.JSON(200, gin.H{"candidates": candidates, "truncated": truncated, "recommended_profile": localImageProfile()})
}

func validateImageCopy(request *imageCopyRequest) error {
	if len(request.Name) > 256 || len(request.Components) < 3 || len(request.Components) > 12 {
		return errors.New("select the three required image components and at most eight LoRAs")
	}
	roles := map[string]bool{}
	loras := 0
	for i := range request.Components {
		component := &request.Components[i]
		if component.Role != "diffusion_model" && component.Role != "text_encoder" && component.Role != "vae" && component.Role != "vision_encoder" && component.Role != "lora" {
			return errors.New("unknown local component role")
		}
		if component.Role == "lora" {
			loras++
			if loras > 8 {
				return errors.New("at most eight LoRAs are supported")
			}
		} else if roles[component.Role] {
			return errors.New("select only one component for each required role")
		}
		roles[component.Role] = true
		if !filepath.IsAbs(component.Path) {
			return errors.New("local component paths must be absolute")
		}
		info, err := os.Stat(component.Path)
		if err != nil || !info.Mode().IsRegular() || info.Size() <= 0 {
			return fmt.Errorf("%s: local component is unavailable", component.Role)
		}
		component.info = info
		inspection, err := imagegen.InspectLocalComponent(component.Path, component.Role)
		if err != nil {
			return fmt.Errorf("%s: %w", component.Role, err)
		}
		if inspection.Compatibility != "supported" {
			return fmt.Errorf("%s: %s", component.Role, inspection.Hint)
		}
		component.supportsEditing = inspection.SupportsEditing
	}
	for _, role := range []string{"diffusion_model", "text_encoder", "vae"} {
		if !roles[role] {
			return fmt.Errorf("missing %s component", role)
		}
	}
	profile := localImageProfile()
	if request.Profile != nil {
		profile = *request.Profile
	}
	// Component selection is authoritative; arbitrary paths in profile settings
	// cannot silently add another adapter or replace a selected component.
	profile.LoRAs = nil
	for _, component := range request.Components {
		if component.Role == "lora" {
			scale := 1.0
			if component.Scale != nil {
				scale = *component.Scale
			}
			profile.LoRAs = append(profile.LoRAs, imagegen.LoRA{Path: component.Path, Scale: scale})
		}
	}
	if err := imagegen.ValidateProfile(profile); err != nil {
		return err
	}
	request.Profile = &profile
	return nil
}

func (s *Server) imageCopyHandler(c *gin.Context) {
	var request imageCopyRequest
	if !imageBind(c, &request) {
		return
	}
	fingerprint := imageFingerprint(request)
	v := s.imageService()
	// Idempotent retries remain valid after the user moves the original files.
	v.mu.Lock()
	for _, existing := range v.jobs {
		if request.RequestID != "" && existing.RequestID == request.RequestID {
			if existing.Fingerprint != fingerprint {
				v.mu.Unlock()
				c.JSON(409, gin.H{"error": "request_id belongs to a different image request"})
				return
			}
			c.JSON(202, gin.H{"job": existing})
			v.mu.Unlock()
			return
		}
	}
	config := cloneImageConfig(v.config)
	v.mu.Unlock()
	if err := validateImageCopy(&request); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	job, err := v.enqueue(&imageJob{Kind: "import", RequestID: request.RequestID, Config: config, Copy: &request, Fingerprint: fingerprint})
	if err != nil {
		c.JSON(409, gin.H{"error": err.Error()})
		return
	}
	v.writeJob(c, http.StatusAccepted, job)
}

// copyImageFile never changes the source. A copied component is published only
// after its complete digest matches a second read of the destination and its
// source identity, size and modification time remain stable.
func copyImageFile(ctx context.Context, source, destination string, expected os.FileInfo, progress func(string, int64, int64)) (string, int64, error) {
	input, err := os.Open(source)
	if err != nil {
		return "", 0, err
	}
	defer input.Close()
	before, err := input.Stat()
	if err != nil {
		return "", 0, err
	}
	stable := func(info os.FileInfo) bool {
		return info != nil && os.SameFile(before, info) && before.Size() == info.Size() && before.ModTime().Equal(info.ModTime())
	}
	if expected != nil && !stable(expected) {
		return "", 0, errors.New("source component changed after selection")
	}
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return "", 0, err
	}
	complete := false
	defer func() {
		output.Close()
		if !complete {
			os.Remove(destination)
		}
	}()
	hash := sha256.New()
	buffer := make([]byte, 2<<20)
	var copied int64
	last := time.Time{}
	for {
		if err := ctx.Err(); err != nil {
			return "", 0, err
		}
		n, readErr := input.Read(buffer)
		if n > 0 {
			if _, err = output.Write(buffer[:n]); err != nil {
				return "", 0, err
			}
			hash.Write(buffer[:n])
			copied += int64(n)
			if progress != nil && time.Since(last) >= 200*time.Millisecond {
				progress("copying", copied, before.Size())
				last = time.Now()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", 0, readErr
		}
	}
	after, err := input.Stat()
	if err != nil || !stable(after) {
		return "", 0, errors.New("source component changed while copying")
	}
	current, err := os.Stat(source)
	if err != nil || !stable(current) {
		return "", 0, errors.New("source path changed while copying")
	}
	if copied != before.Size() {
		return "", 0, errors.New("source size changed while copying")
	}
	if err = output.Sync(); err != nil {
		return "", 0, err
	}
	if err = output.Close(); err != nil {
		return "", 0, err
	}
	digest := hex.EncodeToString(hash.Sum(nil))
	verify, err := os.Open(destination)
	if err != nil {
		return "", 0, err
	}
	defer verify.Close()
	hash.Reset()
	var checked int64
	last = time.Time{}
	for {
		if err := ctx.Err(); err != nil {
			return "", 0, err
		}
		n, readErr := verify.Read(buffer)
		if n > 0 {
			hash.Write(buffer[:n])
			checked += int64(n)
			if progress != nil && time.Since(last) >= 200*time.Millisecond {
				progress("verifying", checked, copied)
				last = time.Now()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", 0, readErr
		}
	}
	if checked != copied || hex.EncodeToString(hash.Sum(nil)) != digest {
		return "", 0, errors.New("copied component SHA256 mismatch")
	}
	current, err = os.Stat(source)
	if err != nil || !stable(current) {
		return "", 0, errors.New("source path changed while verifying the copy")
	}
	complete = true
	return digest, copied, nil
}

func (v *imageService) copyLocal(job *imageJob) error {
	request := job.Copy
	if request == nil || request.Profile == nil {
		return errors.New("missing local copy request")
	}
	id := "local-" + job.ID
	final := filepath.Join(job.Config.ModelsDir, id)
	staging := filepath.Join(job.Config.ModelsDir, ".import-"+job.ID)
	if err := imageWriteAllowed(staging); err != nil {
		return err
	}
	if err := os.MkdirAll(job.Config.ModelsDir, 0700); err != nil {
		return err
	}
	if err := os.Mkdir(staging, 0700); err != nil {
		return err
	}
	created := []string{}
	published := false
	defer func() {
		if !published {
			for _, path := range created {
				os.Remove(path)
			}
			os.Remove(staging)
		}
	}()
	profile := *request.Profile
	profile.ID = "profile-" + job.ID
	profile.ModelID = id
	profile.LoRAs = nil
	profile.DiffusionModel, profile.TextEncoder, profile.VAE, profile.VisionEncoder = "", "", "", ""
	name := strings.TrimSpace(request.Name)
	if name == "" {
		name = "Local Qwen Image 2.1"
	}
	bundle := imagegen.Bundle{ID: id, Name: name, Directory: final, Profile: profile}
	hasVision := false
	for _, component := range request.Components {
		if component.Role == "vision_encoder" {
			hasVision = true
		}
	}
	type receiptComponent struct {
		Role     string `json:"role"`
		Source   string `json:"source"`
		Filename string `json:"filename"`
		SHA256   string `json:"sha256"`
		Bytes    int64  `json:"bytes"`
	}
	receipt := make([]receiptComponent, 0, len(request.Components))
	for i, component := range request.Components {
		filename := fmt.Sprintf("%02d-%s-%s", i+1, component.Role, filepath.Base(component.Path))
		temporary := filepath.Join(staging, filename)
		created = append(created, temporary)
		digest, size, err := copyImageFile(job.ctx, component.Path, temporary, component.info, func(phase string, current, total int64) {
			v.updateDownload(job, imagegen.DownloadProgress{Phase: phase, Role: component.Role, Filename: filepath.Base(component.Path), Message: "Copying and verifying selected local component", Completed: current, Total: total, FileIndex: i + 1, FileCount: len(request.Components)})
		})
		if err != nil {
			return fmt.Errorf("%s: %w", component.Role, err)
		}
		v.updateDownload(job, imagegen.DownloadProgress{Phase: "verified", Role: component.Role, Filename: filepath.Base(component.Path), Message: "Local component copied and SHA256 verified", Completed: size, Total: size, FileIndex: i + 1, FileCount: len(request.Components)})
		receipt = append(receipt, receiptComponent{Role: component.Role, Source: component.Path, Filename: filename, SHA256: digest, Bytes: size})
		target := filepath.Join(final, filename)
		if component.Role == "lora" {
			scale := 1.0
			if component.Scale != nil {
				scale = *component.Scale
			}
			bundle.Profile.LoRAs = append(bundle.Profile.LoRAs, imagegen.LoRA{Path: target, Scale: scale})
			continue
		}
		bundle.Components = append(bundle.Components, imagegen.InstalledComponent{Component: imagegen.Component{Role: component.Role, Filename: filename, SHA256: digest, Size: size, Required: component.Role != "vision_encoder"}, Path: target})
		switch component.Role {
		case "diffusion_model":
			bundle.Profile.DiffusionModel = target
		case "text_encoder":
			bundle.Profile.TextEncoder = target
			if component.supportsEditing && !hasVision {
				bundle.Profile.VisionEncoder = target
				bundle.Components = append(bundle.Components, imagegen.InstalledComponent{Component: imagegen.Component{Role: "vision_encoder", Filename: filename, SHA256: digest, Size: size, Required: false}, Path: target})
			}
		case "vae":
			bundle.Profile.VAE = target
		case "vision_encoder":
			bundle.Profile.VisionEncoder = target
		}
	}
	if err := job.ctx.Err(); err != nil {
		return err
	}
	manifest := filepath.Join(staging, imagegen.ManifestName)
	created = append(created, manifest)
	if err := writeImageJSON(manifest, bundle); err != nil {
		return err
	}
	receiptPath := filepath.Join(staging, "local-copy-verification.json")
	created = append(created, receiptPath)
	if err := writeImageJSON(receiptPath, receipt); err != nil {
		return err
	}
	if err := os.Rename(staging, final); err != nil {
		return err
	}
	published = true
	// The catalog validates the same public manifest and final component paths.
	models, _, err := imagegen.Catalog(job.Config.ModelsDir)
	if err != nil {
		return err
	}
	for _, model := range models {
		if model.ID == id {
			if !model.Ready {
				return fmt.Errorf("copied pipeline is incomplete: %s", strings.Join(model.Missing, "; "))
			}
			if err := v.recordBundle(model); err != nil {
				return err
			}
			v.mu.Lock()
			job.ModelID = id
			job.ProfileID = model.Profile.ID
			v.mu.Unlock()
			return nil
		}
	}
	return errors.New("copied pipeline was not found in the model catalog")
}
