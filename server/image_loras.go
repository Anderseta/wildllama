package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
	"github.com/ollama/ollama/types/model"
)

type imageLoRAFile struct {
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Filename   string `json:"filename"`
	Size       int64  `json:"size"`
	SHA256     string `json:"sha256"`
}

type imageLoRAFiles struct {
	Repository string          `json:"repository"`
	Revision   string          `json:"revision"`
	Files      []imageLoRAFile `json:"files"`
	Truncated  bool            `json:"truncated,omitempty"`
}

type imageLoRAInstallRequest struct {
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Filename   string `json:"filename"`
	SHA256     string `json:"sha256,omitempty"`
	RequestID  string `json:"request_id,omitempty"`
}

func validateImageLoRASource(repository, revision, filename string, requireFile bool) error {
	if len(repository) > 256 || !huggingFaceRepositoryID.MatchString(repository) || strings.Contains(repository, "..") {
		return errors.New("use a Hugging Face repository in owner/model format")
	}
	if revision != "" && !huggingFaceRevision.MatchString(revision) || requireFile && revision == "" {
		return errors.New("select an immutable Hugging Face commit revision")
	}
	if requireFile {
		if err := validateCreateFilePath(filename); err != nil || strings.Contains(filename, "\\") || !strings.HasSuffix(strings.ToLower(filename), ".safetensors") {
			return errors.New("select a safetensors LoRA filename from the repository file list")
		}
	}
	return nil
}

func imageLoRAFilesFromRepository(repository string, source huggingFaceRepository) (imageLoRAFiles, error) {
	result := imageLoRAFiles{Repository: repository, Revision: source.SHA, Files: []imageLoRAFile{}}
	if !huggingFaceRevision.MatchString(source.SHA) {
		return result, errors.New("Hugging Face did not provide a pinned repository revision")
	}
	for _, file := range source.Files {
		file = normalizeHuggingFaceFile(file)
		if !strings.HasSuffix(strings.ToLower(file.Filename), ".safetensors") || strings.Contains(file.Filename, "\\") {
			continue
		}
		if validateCreateFilePath(file.Filename) != nil || validateHuggingFaceFile(file) != nil {
			continue
		}
		if len(result.Files) >= 512 {
			result.Truncated = true
			break
		}
		result.Files = append(result.Files, imageLoRAFile{Repository: repository, Revision: source.SHA, Filename: file.Filename, Size: file.Size, SHA256: strings.ToLower(file.LFS.SHA256)})
	}
	sort.Slice(result.Files, func(i, j int) bool { return result.Files[i].Filename < result.Files[j].Filename })
	return result, nil
}

func resolveImageLoRA(ctx context.Context, repository, revision, filename string) (imagegen.Component, string, error) {
	if err := validateImageLoRASource(repository, revision, filename, true); err != nil {
		return imagegen.Component{}, "", err
	}
	options := &registryOptions{Token: os.Getenv("HF_TOKEN"), HuggingFace: huggingFaceSelection{Revision: revision}}
	source, err := readHuggingFaceRepository(ctx, model.ParseName("huggingface.co/"+repository), options)
	if err != nil {
		return imagegen.Component{}, "", err
	}
	files, err := imageLoRAFilesFromRepository(repository, source)
	if err != nil {
		return imagegen.Component{}, "", err
	}
	for _, file := range files.Files {
		if file.Filename == filename {
			return imagegen.Component{Role: "lora", Repo: repository, Revision: files.Revision, Filename: filename, SHA256: file.SHA256, Size: file.Size, Required: true}, options.Token, nil
		}
	}
	return imagegen.Component{}, "", errors.New("selected LoRA file is absent or has no verified SHA256/size metadata")
}

func (s *Server) imageLoRAFilesHandler(c *gin.Context) {
	repository, revision := strings.TrimSpace(c.Query("repository")), strings.TrimSpace(c.Query("revision"))
	if err := validateImageLoRASource(repository, revision, "", false); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	ctx, cancel := context.WithTimeout(c.Request.Context(), 20*time.Second)
	defer cancel()
	data, err := huggingFaceMetadata.get(ctx, "image-lora-files:"+repository+"@"+revision, time.Minute, func(ctx context.Context) ([]byte, error) {
		source, err := readHuggingFaceRepository(ctx, model.ParseName("huggingface.co/"+repository), &registryOptions{Token: os.Getenv("HF_TOKEN"), HuggingFace: huggingFaceSelection{Revision: revision}})
		if err != nil {
			return nil, err
		}
		result, err := imageLoRAFilesFromRepository(repository, source)
		if err != nil {
			return nil, err
		}
		return json.Marshal(result)
	})
	if err != nil {
		huggingFaceCatalogError(c, err)
		return
	}
	c.Data(http.StatusOK, "application/json; charset=utf-8", data)
}

func (s *Server) imageLoRAInstallHandler(c *gin.Context) {
	var request imageLoRAInstallRequest
	if !imageBind(c, &request) {
		return
	}
	if err := validateImageLoRASource(request.Repository, request.Revision, request.Filename, true); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	v := s.imageService()
	fingerprint := imageFingerprint(request)
	v.mu.Lock()
	for _, existing := range v.jobs {
		if request.RequestID != "" && existing.RequestID == request.RequestID {
			if existing.Fingerprint != fingerprint {
				v.mu.Unlock()
				c.JSON(409, gin.H{"error": "request_id belongs to a different image request"})
				return
			}
			response := cloneImageJobLocked(existing)
			v.mu.Unlock()
			c.JSON(202, gin.H{"job": response})
			return
		}
	}
	config := cloneImageConfig(v.config)
	v.mu.Unlock()
	ctx, cancel := context.WithTimeout(c.Request.Context(), 20*time.Second)
	defer cancel()
	component, _, err := resolveImageLoRA(ctx, request.Repository, request.Revision, request.Filename)
	if err != nil {
		huggingFaceCatalogError(c, err)
		return
	}
	if request.SHA256 != "" && !strings.EqualFold(request.SHA256, component.SHA256) {
		c.JSON(400, gin.H{"error": "selected LoRA checksum changed; refresh available files"})
		return
	}
	job, err := v.enqueue(&imageJob{Kind: "install_lora", RequestID: request.RequestID, Config: config, LoRA: &component, Fingerprint: fingerprint})
	if err != nil {
		c.JSON(409, gin.H{"error": err.Error()})
		return
	}
	v.writeJob(c, 202, job)
}

func (v *imageService) installLoRA(job *imageJob) error {
	if job.LoRA == nil {
		return errors.New("missing LoRA download selection")
	}
	expected := *job.LoRA
	metadataCtx, cancelMetadata := context.WithTimeout(job.ctx, 20*time.Second)
	component, token, err := resolveImageLoRA(metadataCtx, expected.Repo, expected.Revision, expected.Filename)
	cancelMetadata()
	if err != nil {
		return err
	}
	if component != expected {
		return errors.New("pinned LoRA metadata changed; refresh available files")
	}
	ctx := imagegen.WithDownloadToken(job.ctx, token)
	asset, err := imagegen.InstallLoRA(ctx, job.Config.ModelsDir, component, func(progress imagegen.DownloadProgress) { v.updateDownload(job, progress) })
	if err != nil {
		return err
	}
	v.mu.Lock()
	job.ModelID = asset.ID
	v.mu.Unlock()
	return nil
}

func (s *Server) imageLoRAsHandler(c *gin.Context) {
	v := s.imageService()
	v.mu.Lock()
	config := cloneImageConfig(v.config)
	v.mu.Unlock()
	assets, err := imagegen.LoRALibrary(config.ModelsDir)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	seen := map[string]bool{}
	normalized := func(path string) string {
		path = filepath.Clean(path)
		if filepath.VolumeName(path) != "" {
			path = strings.ToLower(path)
		}
		return path
	}
	for _, asset := range assets {
		seen[normalized(asset.Path)] = true
	}
	add := func(lora imagegen.LoRA, origin string) {
		key := normalized(lora.Path)
		if lora.Path == "" || seen[key] || len(assets) >= 1024 {
			return
		}
		seen[key] = true
		sum := sha256.Sum256([]byte(key))
		asset := imagegen.LoRAAsset{ID: "local-" + hex.EncodeToString(sum[:12]), Name: filepath.Base(lora.Path), Path: lora.Path, Origin: origin}
		info, err := os.Stat(lora.Path)
		if err == nil && info.Mode().IsRegular() {
			asset.Size = info.Size()
			inspection, inspectErr := imagegen.InspectLocalComponent(lora.Path, "lora")
			asset.Ready = inspectErr == nil && inspection.Compatibility == "supported"
			if inspectErr != nil {
				asset.Error = inspectErr.Error()
			} else if !asset.Ready {
				asset.Error = inspection.Hint
			}
		} else {
			asset.Error = "local LoRA file is unavailable"
		}
		assets = append(assets, asset)
	}
	models, _, err := imagegen.Catalog(config.ModelsDir)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	for _, bundle := range models {
		for _, lora := range bundle.Profile.LoRAs {
			add(lora, "local")
		}
	}
	for _, profile := range config.Profiles {
		for _, lora := range profile.LoRAs {
			add(lora, "local")
		}
	}
	sort.Slice(assets, func(i, j int) bool { return strings.ToLower(assets[i].Name) < strings.ToLower(assets[j].Name) })
	if assets == nil {
		assets = []imagegen.LoRAAsset{}
	}
	c.JSON(200, gin.H{"loras": assets})
}
