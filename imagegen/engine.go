package imagegen

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/internal/cudaload"
	_ "golang.org/x/image/webp"
)

type Engine struct {
	config Config
	mu     sync.Mutex
	closed bool
	pid    int
	cancel context.CancelFunc
	done   chan struct{}
}

type imageMemoryMode int

const (
	imageGPUResident imageMemoryMode = iota
	imageRAMGPU
	imageRAMCPU
)

// NewEngine is read-only: status/configuration requests never create a cache,
// start a process, or download weights. Write directories are created by Run.
func NewEngine(config Config) (*Engine, error) {
	if config.MinFreeDiskMiB < 0 || config.MinFreeDiskMiB > 1024*1024 {
		return nil, errors.New("invalid minimum free disk space")
	}
	for name, path := range map[string]string{"runner_path": config.RunnerPath, "work_dir": config.WorkDir, "models_dir": config.ModelsDir, "output_dir": config.OutputDir} {
		if path == "" || !filepath.IsAbs(path) {
			return nil, fmt.Errorf("image %s must be an absolute path", name)
		}
		if name != "runner_path" {
			if err := validateWritePath(path); err != nil {
				return nil, err
			}
		}
	}
	return &Engine{config: config}, nil
}

func (e *Engine) Status() RunnerStatus {
	e.mu.Lock()
	defer e.mu.Unlock()
	status := RunnerStatus{RunnerPath: e.config.RunnerPath, Revision: "unknown", Running: e.done != nil, PID: e.pid}
	if e.closed {
		status.Error = "image engine is closed"
		return status
	}
	info, err := os.Stat(e.config.RunnerPath)
	if err != nil {
		status.Error = "image runner is not installed: " + err.Error()
		return status
	}
	if !info.Mode().IsRegular() {
		status.Error = "image runner must be a regular executable"
		return status
	}
	manifestPath := filepath.Join(filepath.Dir(e.config.RunnerPath), "wildimage-runtime.json")
	if _, err := os.Stat(manifestPath); err == nil {
		var manifest struct {
			Revision string `json:"revision"`
			Features struct {
				ConditioningCache bool `json:"conditioning_cache"`
				RAMBudget         bool `json:"ram_budget"`
				MOSSEdit          bool `json:"moss_edit"`
				InpaintCrop       bool `json:"inpaint_crop"`
			} `json:"features"`
			Files []struct {
				Name   string `json:"name"`
				Size   int64  `json:"size"`
				SHA256 string `json:"sha256"`
			} `json:"files"`
		}
		f, err := os.Open(manifestPath)
		if err != nil {
			status.Error = err.Error()
			return status
		}
		data, err := io.ReadAll(io.LimitReader(f, 1<<20))
		f.Close()
		if err != nil || json.Unmarshal([]byte(strings.TrimPrefix(string(data), "\ufeff")), &manifest) != nil {
			status.Error = "invalid image runtime manifest"
			return status
		}
		var executableHash string
		for _, file := range manifest.Files {
			if filepath.Base(file.Name) != file.Name {
				status.Error = "invalid runtime library path"
				return status
			}
			fileInfo, err := os.Stat(filepath.Join(filepath.Dir(e.config.RunnerPath), file.Name))
			if err != nil || !fileInfo.Mode().IsRegular() || fileInfo.Size() != file.Size {
				status.Error = "missing or changed isolated runtime file: " + file.Name
				return status
			}
			if strings.EqualFold(file.Name, filepath.Base(e.config.RunnerPath)) {
				executableHash = file.SHA256
			}
		}
		if executableHash == "" {
			status.Error = "runtime manifest does not include this executable"
			return status
		}
		digest, err := conditioningFileIdentity(context.Background(), e.config.RunnerPath, true)
		if err != nil {
			status.Error = err.Error()
			return status
		}
		if !strings.EqualFold(digest, executableHash) {
			status.Error = "image runner SHA256 differs from its pinned manifest"
			return status
		}
		status.Revision = manifest.Revision
		status.ConditioningCacheSupported = manifest.Features.ConditioningCache
		status.RAMBudgetSupported = manifest.Features.RAMBudget
		status.MOSSEditSupported = manifest.Features.MOSSEdit
		status.InpaintCropSupported = manifest.Features.InpaintCrop
	}
	if !status.RAMBudgetSupported {
		status.Error = "image runtime does not support bounded RAM offload; update the isolated image runtime"
		return status
	}
	status.Ready = true
	return status
}

func (e *Engine) Close() error {
	e.mu.Lock()
	e.closed = true
	cancel, done := e.cancel, e.done
	e.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}
	return nil
}

func ValidateProfile(p Profile) error {
	if p.Width < 64 || p.Width > 4096 || p.Width%32 != 0 || p.Height < 64 || p.Height > 4096 || p.Height%32 != 0 {
		return errors.New("image dimensions must be multiples of 32 between 64 and 4096")
	}
	if p.Steps < 1 || p.Steps > 100 {
		return errors.New("image steps must be between 1 and 100")
	}
	if math.IsNaN(p.CFGScale) || math.IsInf(p.CFGScale, 0) || p.CFGScale < 0 || p.CFGScale > 30 {
		return errors.New("image CFG must be finite and between 0 and 30")
	}
	if p.Sampler != "euler" {
		return errors.New("this image pipeline currently supports the verified Euler sampler")
	}
	if p.Scheduler != "" && p.Scheduler != "simple" && p.Scheduler != "discrete" && p.Scheduler != "karras" {
		return errors.New("unsupported image scheduler")
	}
	if p.Tokenizer != "" && !isTokenizerPath(p.Tokenizer) {
		return errors.New("image tokenizer must be an absolute JSON path")
	}
	if len(p.LoRAs) > 8 {
		return errors.New("at most eight image LoRAs are supported")
	}
	if p.MOSSEditLoRA != nil {
		edit := *p.MOSSEditLoRA
		if p.ModelID != MOSSRecipeID || !filepath.IsAbs(edit.Path) ||
			strings.ContainsAny(edit.Path, "\x00\r\n") || !strings.EqualFold(filepath.Ext(edit.Path), ".safetensors") ||
			math.IsNaN(edit.Scale) || math.IsInf(edit.Scale, 0) || edit.Scale < 0 || edit.Scale > 4 {
			return errors.New("MOSS edit requires an absolute safetensors adapter path and a strength between 0 and 4")
		}
		if len(p.LoRAs) >= 8 && p.UsesMOSSEdit() {
			return errors.New("the MOSS edit adapter uses one of the eight available LoRA slots")
		}
		for _, lora := range p.LoRAs {
			if sameComponentPath(edit.Path, lora.Path) {
				return errors.New("select the MOSS edit adapter only in its dedicated slot")
			}
		}
	}
	for _, lora := range p.LoRAs {
		if math.IsNaN(lora.Scale) || math.IsInf(lora.Scale, 0) || lora.Scale < -4 || lora.Scale > 4 {
			return errors.New("LoRA scale must be finite and between -4 and 4")
		}
		if lora.Scale != 0 && (!filepath.IsAbs(lora.Path) || strings.ContainsAny(lora.Path, "\x00\r\n")) {
			return errors.New("active LoRA paths must be absolute")
		}
	}
	if p.MaxVRAMMiB < 0 || p.MaxVRAMMiB > 1048576 {
		return errors.New("image VRAM fitting target must be between 0 and 1048576 MiB")
	}
	if p.MaxRAMMiB < 0 || p.MaxRAMMiB > 1048576 {
		return errors.New("image RAM budget must be between 0 (bounded Auto) and 1048576 MiB")
	}
	if p.VRAMReserveMiB < 0 || p.VRAMReserveMiB > 1048576 {
		return errors.New("image free VRAM margin must be between 0 and 1048576 MiB")
	}
	if p.CPUThreads < 0 || p.CPUThreads > 1024 {
		return errors.New("image CPU threads must be between 0 and 1024")
	}
	if p.Sigmas != "" {
		values := strings.Split(p.Sigmas, ",")
		if len(values) != p.Steps+1 {
			return errors.New("sigma count must equal steps + 1, including final zero")
		}
		previous := math.Inf(1)
		for i, value := range values {
			n, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
			if err != nil || math.IsNaN(n) || math.IsInf(n, 0) || n < 0 || n > previous {
				return errors.New("sigmas must be finite, nonnegative and descending")
			}
			if i == len(values)-1 && n != 0 {
				return errors.New("the sigma schedule must end in zero")
			}
			previous = n
		}
	}
	return nil
}

func ValidateRequest(p Profile, r Request) error {
	_, _, err := prepareImageRequest(p, r)
	return err
}

func prepareImageRequest(p Profile, r Request) (Profile, *image.Gray, error) {
	var mask *image.Gray
	var err error
	if r.Inpaint != nil {
		if len(r.ReferenceImages) == 0 {
			return p, nil, errors.New("inpaint requires a reference")
		}
		mask, err = r.Inpaint.bitmap()
		if err != nil {
			return p, nil, err
		}
	}
	if r.CPUOnly && !p.OffloadToCPU {
		return p, nil, errors.New("CPU image admission requires enabled RAM offload")
	}
	p, err = ResolveProfileDimensions(p, r)
	if err != nil {
		return p, nil, err
	}
	return p, mask, validateImageInputs(p, r)
}

func validateImageInputs(p Profile, r Request) error {
	if err := ValidateProfile(p); err != nil {
		return err
	}
	if strings.TrimSpace(r.Prompt) == "" || len(r.Prompt) > 32768 || len(r.NegativePrompt) > 32768 {
		return errors.New("image prompt must contain 1–32768 bytes; negative prompt is limited to 32768 bytes")
	}
	if strings.IndexByte(r.Prompt, 0) >= 0 || strings.IndexByte(r.NegativePrompt, 0) >= 0 {
		return errors.New("image prompts cannot contain NUL")
	}
	if strings.Contains(strings.ToLower(r.Prompt), "<lora:") || strings.Contains(strings.ToLower(r.NegativePrompt), "<lora:") {
		return errors.New("select LoRAs explicitly in the image profile instead of embedding loader tags in the prompt")
	}
	for role, path := range map[string]string{"diffusion model": p.DiffusionModel, "text encoder": p.TextEncoder, "VAE": p.VAE} {
		if err := validateModelFile(path); err != nil {
			return fmt.Errorf("%s: %w", role, err)
		}
	}
	if p.Tokenizer != "" || p.ModelID == MOSSRecipeID {
		if err := validateTokenizerFile(p.Tokenizer); err != nil {
			return fmt.Errorf("tokenizer: %w", err)
		}
	}
	if len(r.ReferenceImages) > 4 {
		return errors.New("at most four reference images are supported per job")
	}
	if len(r.ReferenceImages) > 0 && !p.UsesMOSSEdit() {
		if p.ModelID == MOSSRecipeID {
			return errors.New("MOSS reference editing requires an active MOSS edit LoRA")
		}
		if err := validateModelFile(p.VisionEncoder); err != nil {
			return fmt.Errorf("reference edits require a valid vision projector: %w", err)
		}
		if sameComponentPath(p.VisionEncoder, p.TextEncoder) {
			inspection, err := InspectLocalComponent(p.TextEncoder, "text_encoder")
			if err != nil {
				return err
			}
			if !inspection.SupportsEditing {
				return errors.New("the shared encoder file has no recognized embedded vision tower")
			}
		}
	}
	for _, lora := range p.requestLoRAs(r.ReferenceImages) {
		if lora.Scale == 0 {
			continue
		}
		inspection, err := InspectLocalComponent(lora.Path, "lora")
		if err != nil {
			return fmt.Errorf("LoRA: %w", err)
		}
		if inspection.Compatibility != "supported" {
			return errors.New("LoRA: " + inspection.Hint)
		}
		if p.UsesMOSSEdit() && sameComponentPath(lora.Path, p.MOSSEditLoRA.Path) && !inspection.MOSSEdit {
			return errors.New("the MOSS edit adapter must carry moss-edit-lora metadata")
		}
	}
	for _, path := range r.ReferenceImages {
		if _, err := referenceImageConfig(path); err != nil {
			return err
		}
	}
	return nil
}

func validateModelFile(path string) error {
	f, info, err := openModelFile(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return validateModelHeader(f, info)
}

func openModelFile(path string) (*os.File, os.FileInfo, error) {
	if !filepath.IsAbs(path) {
		return nil, nil, errors.New("model component path must be absolute")
	}
	ext := strings.ToLower(filepath.Ext(path))
	if ext != ".gguf" && ext != ".safetensors" {
		return nil, nil, errors.New("expected a GGUF or safetensors component")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() < 16 {
		f.Close()
		return nil, nil, errors.New("model component is missing or truncated")
	}
	return f, info, nil
}

func validateModelHeader(f *os.File, info os.FileInfo) error {
	var header [8]byte
	if _, err := f.ReadAt(header[:], 0); err != nil {
		return err
	}
	ext := strings.ToLower(filepath.Ext(f.Name()))
	if ext == ".gguf" && string(header[:4]) != "GGUF" {
		return errors.New("invalid GGUF header")
	}
	if ext == ".safetensors" {
		n := binary.LittleEndian.Uint64(header[:])
		if n < 2 || n > 256<<20 || n > uint64(info.Size()-8) {
			return errors.New("invalid safetensors header length")
		}
	}
	return nil
}

func isTokenizerPath(path string) bool {
	return filepath.IsAbs(path) && strings.EqualFold(filepath.Ext(path), ".json") &&
		!strings.ContainsAny(path, "\x00\r\n,=")
}

func validateTokenizerFile(path string) error {
	if !isTokenizerPath(path) {
		return errors.New("expected an absolute tokenizer JSON path without comma or equals sign")
	}
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() < 2 || info.Size() > 64<<20 {
		return errors.New("tokenizer JSON is missing or outside the 64 MiB limit")
	}
	// The runner parses and checks the full vocabulary when loading it.
	for reader := bufio.NewReader(file); ; {
		value, err := reader.ReadByte()
		if err != nil {
			return errors.New("tokenizer JSON is empty")
		}
		switch value {
		case ' ', '\r', '\n', '\t':
			continue
		case '{':
			return nil
		default:
			return errors.New("tokenizer JSON must contain an object")
		}
	}
}

func validateWritePath(path string) error {
	if !filepath.IsAbs(path) {
		return errors.New("image write paths must be absolute")
	}
	resolved, err := resolvedWritePath(path)
	if err != nil {
		return err
	}
	// Resolve the nearest existing ancestor before creating anything. This also
	// rejects a G: junction leading to the system disk.
	if runtime.GOOS == "windows" && strings.EqualFold(filepath.VolumeName(resolved), "C:") {
		return errors.New("image files and caches must be stored outside C:")
	}
	return nil
}

func resolvedWritePath(path string) (string, error) {
	path = filepath.Clean(path)
	for ancestor := path; ; ancestor = filepath.Dir(ancestor) {
		if resolved, err := filepath.EvalSymlinks(ancestor); err == nil {
			suffix, err := filepath.Rel(ancestor, path)
			if err != nil {
				return "", err
			}
			return filepath.Join(resolved, suffix), nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
		if filepath.Dir(ancestor) == ancestor {
			return path, nil
		}
	}
}

func within(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

func (e *Engine) Run(ctx context.Context, p Profile, r Request, progress func(Progress)) (Result, error) {
	ctx = WithDiskReserve(ctx, e.config.MinFreeDiskMiB)
	runnerStatus := e.Status()
	if !runnerStatus.Ready {
		return Result{}, errors.New(runnerStatus.Error)
	}
	if p.UsesMOSSEdit() && len(r.ReferenceImages) > 0 && !runnerStatus.MOSSEditSupported {
		return Result{}, errors.New("the image runner does not support MOSS edits")
	}
	p, mask, err := prepareImageRequest(p, r)
	if err != nil {
		return Result{}, err
	}
	var original image.Point
	if r.Inpaint != nil && r.Inpaint.Crop {
		if !runnerStatus.InpaintCropSupported {
			return Result{}, errors.New("the installed image runner does not support cropped inpainting")
		}
		cfg, err := referenceImageConfig(r.ReferenceImages[0])
		if err != nil {
			return Result{}, err
		}
		original = image.Pt(cfg.Width, cfg.Height)
		var canvas image.Point
		r.crop, canvas = planInpaintCrop(mask, original, image.Pt(p.Width, p.Height))
		p.Width, p.Height = canvas.X, canvas.Y
	}
	output := filepath.Clean(r.OutputPath)
	if !filepath.IsAbs(output) || !within(filepath.Clean(e.config.OutputDir), output) || strings.ToLower(filepath.Ext(output)) != ".png" || strings.Contains(filepath.Base(output), "%") {
		return Result{}, errors.New("image output must be a PNG path inside the configured output directory")
	}
	if err := validateWritePath(output); err != nil {
		return Result{}, err
	}
	rootResolved, err := resolvedWritePath(e.config.OutputDir)
	if err != nil {
		return Result{}, err
	}
	outputResolved, err := resolvedWritePath(output)
	if err != nil {
		return Result{}, err
	}
	if !within(rootResolved, outputResolved) {
		return Result{}, errors.New("image output resolves outside the configured output directory")
	}
	if _, err := os.Lstat(output); err == nil {
		return Result{}, errors.New("image output already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return Result{}, err
	}
	if err := CheckDiskSpace(ctx, output, int64(p.Width)*int64(p.Height)*4+(16<<20)); err != nil {
		return Result{}, err
	}
	if err := CheckDiskSpace(ctx, e.config.WorkDir, 16<<20); err != nil {
		return Result{}, err
	}
	ctx, cancel := context.WithCancel(ctx)
	var diskGuard <-chan error
	if e.config.MinFreeDiskMiB > 0 {
		diskGuard = monitorImageDisks(ctx, cancel, []string{e.config.WorkDir, filepath.Dir(output)})
	}
	e.mu.Lock()
	if e.closed || e.done != nil {
		e.mu.Unlock()
		cancel()
		return Result{}, errors.New("image engine is closed or already running")
	}
	done := make(chan struct{})
	e.done = done
	e.cancel = cancel
	e.mu.Unlock()
	defer func() {
		cancel()
		e.mu.Lock()
		e.done = nil
		e.cancel = nil
		e.pid = 0
		close(done)
		e.mu.Unlock()
	}()
	for _, dir := range []string{e.config.WorkDir, filepath.Dir(output)} {
		if err := os.MkdirAll(dir, 0700); err != nil {
			return Result{}, err
		}
	}
	for _, sub := range []string{"cuda-cache", "cache", "hf/hub", "hf/xet", "local", "appdata", "home"} {
		if err := os.MkdirAll(filepath.Join(e.config.WorkDir, sub), 0700); err != nil {
			return Result{}, err
		}
	}
	jobDir, err := os.MkdirTemp(e.config.WorkDir, "job-")
	if err != nil {
		return Result{}, err
	}
	defer os.RemoveAll(jobDir)
	finalPath := output
	if r.Inpaint != nil {
		if err := writeInpaintMask(mask, jobDir, p.Width, p.Height, r.crop, original); err != nil {
			return Result{}, err
		}
		output = filepath.Join(jobDir, "generated.png")
	}
	loraTags, err := prepareLoRAs(ctx, p.requestLoRAs(r.ReferenceImages), jobDir)
	if err != nil {
		return Result{}, err
	}
	prompt, negativePrompt := runnerReferencePrompts(p.DiffusionModel, r)
	if err := os.WriteFile(filepath.Join(jobDir, "prompt.txt"), []byte(prompt+loraTags), 0600); err != nil {
		return Result{}, err
	}
	if err := os.WriteFile(filepath.Join(jobDir, "negative.txt"), []byte(negativePrompt), 0600); err != nil {
		return Result{}, err
	}
	seed := r.Seed
	if seed < 0 {
		var data [8]byte
		if _, err := rand.Read(data[:]); err != nil {
			return Result{}, err
		}
		seed = int64(binary.LittleEndian.Uint64(data[:]) & math.MaxInt64)
	}
	start := time.Now()
	log := &progressLog{callback: progress, start: start, expectedSteps: p.Steps}
	if !r.crop.Empty() {
		log.emit(Progress{Phase: "preparing", Message: fmt.Sprintf("Editando recorte de %d × %d pixels em %d × %d. A saída mantém as dimensões da referência.", r.crop.Dx(), r.crop.Dy(), p.Width, p.Height)}, true)
	}
	mode := imageGPUResident
	residentFailedAt := 0
	if r.CPUOnly {
		mode = imageRAMCPU
	} else if p.OffloadToCPU && p.MaxVRAMMiB > 0 && p.MaxVRAMMiB <= r.GPUResidentFailedAtMiB {
		mode = imageRAMGPU
		residentFailedAt = r.GPUResidentFailedAtMiB
		log.emit(Progress{Phase: "ram_offload", Message: "Reutilizando a execução em GPU com apoio da RAM, validada neste lote."}, true)
	}
	var runErr error
	var decoded image.Image
	eagerCUDA := false
	allowCUDARetry := runtime.GOOS == "windows" && !cudaload.HasExplicitModuleLoading(os.Environ())
	for {
		err = e.runChild(ctx, p, r, jobDir, output, seed, mode, eagerCUDA, runnerStatus, log)
		select {
		case diskErr := <-diskGuard:
			runErr = diskErr
		default:
		}
		var exit *exec.ExitError
		if runErr != nil || ctx.Err() != nil || mode == imageRAMCPU || !errors.As(err, &exit) {
			break
		}
		retryCUDA := allowCUDARetry && !eagerCUDA && cudaload.IsSharedObjectInitFailure(log.tail())
		retryMemory := p.OffloadToCPU && exit.ExitCode() == GPUOutOfMemoryExitCode
		if !retryCUDA && !retryMemory {
			break
		}
		decoded, _ = decodeOutputPNG(output, p.Width, p.Height)
		if decoded != nil {
			break
		}
		// The failed process and its CPU/GPU resources have been fully closed.
		// A saved PNG, cancellation, and unrelated errors must never trigger a
		// second generation. Both recoveries keep the seed and image settings.
		if removeErr := os.Remove(output); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			return Result{}, fmt.Errorf("remove incomplete GPU attempt: %w", removeErr)
		}
		if retryCUDA {
			eagerCUDA = true
			log.emit(Progress{Phase: "cuda_module_retry", Message: "O CUDA falhou ao inicializar um kernel. Tentando novamente na GPU com carregamento antecipado e a mesma seed."}, true)
			continue
		}
		if mode == imageGPUResident {
			residentFailedAt = p.MaxVRAMMiB
			mode = imageRAMGPU
			log.emit(Progress{Phase: "ram_offload", Message: "Os pesos e buffers não couberam na GPU; usando RAM dentro do orçamento e mantendo o cálculo na GPU."}, true)
		} else {
			mode = imageRAMCPU
			log.emit(Progress{Phase: "cpu_fallback", Message: "A GPU não comportou o cálculo com offload; tentando na CPU dentro do orçamento de RAM."}, true)
		}
	}
	if runErr != nil {
		// Keep the actionable disk reason instead of reporting a generic cancel.
	} else if ctx.Err() != nil {
		runErr = ctx.Err()
	} else if err != nil {
		runErr = err
	}
	if decoded == nil {
		log.emit(Progress{Phase: "validating_output", Message: "Verificando o PNG salvo"}, true)
		var decodeErr error
		decoded, decodeErr = decodeOutputPNG(output, p.Width, p.Height)
		if decodeErr != nil {
			_ = os.Remove(output)
			if runErr != nil {
				return Result{}, runErr
			}
			return Result{}, fmt.Errorf("runner did not produce a complete PNG with the requested dimensions: %w", decodeErr)
		}
	}
	if r.Inpaint != nil {
		if runErr != nil {
			return Result{}, runErr
		}
		size, err := composeInpaint(ctx, mask, r.ReferenceImages[0], decoded, finalPath, r.crop)
		if err != nil {
			return Result{}, err
		}
		p.Width, p.Height = size.X, size.Y
		output = finalPath
	}
	result := Result{Path: output, MIMEType: "image/png", Width: p.Width, Height: p.Height, Seed: seed, DurationSeconds: time.Since(start).Seconds(), RunnerRevision: runnerStatus.Revision}
	// A process can fail, or cancellation can arrive, immediately after it saves
	// the complete image. Return both facts so the caller can preserve the asset
	// while still reporting cancellation/failure and restoring the chat model.
	if runErr != nil {
		return result, runErr
	}
	if mode == imageRAMGPU {
		result.GPUResidentFailedAtMiB = residentFailedAt
	}
	log.emit(Progress{Phase: "completed", Message: "Image saved", Current: p.Steps, Total: p.Steps}, true)
	return result, nil
}

func (e *Engine) runChild(ctx context.Context, p Profile, r Request, jobDir, output string, seed int64, mode imageMemoryMode, eagerCUDA bool, status RunnerStatus, log *progressLog) error {
	args := buildArgumentsForMode(p, r, jobDir, output, seed, mode)
	if mode == imageRAMCPU {
		// CPU conditioning gets a separate exact cache namespace. The saved
		// profile, generation summary and user-selected resource caps stay intact.
		p.EncoderOnCPU, p.VAEOnCPU = true, true
	}
	cmd := exec.CommandContext(ctx, e.config.RunnerPath, args...)
	cmd.Dir = jobDir
	cmd.Env = isolatedEnvironment(e.config, jobDir)
	if eagerCUDA && mode != imageRAMCPU {
		cmd.Env = append(cmd.Env, cudaload.ModuleLoadingEnv+"=EAGER")
	}
	if status.ConditioningCacheSupported && e.config.ConditioningCacheMiB > 0 {
		cacheEnv, cacheErr := conditioningCacheEnvironment(ctx, e.config, p, r, status.Revision, log.callback)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if cacheErr == nil {
			cmd.Env = append(cmd.Env, cacheEnv...)
		}
	}
	prepareCommand(cmd)
	cmd.Stdout, cmd.Stderr = log, log
	var guardMu sync.Mutex
	var guard io.Closer
	cmd.Cancel = func() error {
		guardMu.Lock()
		defer guardMu.Unlock()
		if guard != nil {
			_ = guard.Close()
		}
		if cmd.Process != nil {
			return cmd.Process.Kill()
		}
		return nil
	}
	cmd.WaitDelay = 5 * time.Second
	message := "Starting the isolated CUDA image runner"
	if mode == imageRAMCPU {
		message = "Iniciando a tentativa na CPU com os mesmos parâmetros de imagem e orçamento de RAM."
	} else if eagerCUDA {
		message = "Reiniciando o gerador com carregamento antecipado dos kernels CUDA."
	}
	log.mu.Lock()
	log.counterPhase, log.loadPhase = "", ""
	log.lines = log.lines[:0]
	log.emit(Progress{Phase: "starting", Message: message}, true)
	log.mu.Unlock()
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start image runner: %w", err)
	}
	peak, closePeak := newImageWorkingSet(cmd.Process.Pid)
	if closePeak != nil {
		defer closePeak()
	}
	e.mu.Lock()
	e.pid = cmd.Process.Pid
	e.mu.Unlock()
	log.mu.Lock()
	log.pid = cmd.Process.Pid
	log.peakWorkingSet = peak
	log.mu.Unlock()
	defer func() {
		e.mu.Lock()
		e.pid = 0
		e.mu.Unlock()
		log.mu.Lock()
		log.sampleWorkingSet()
		log.peakWorkingSet = nil
		log.pid = 0
		log.mu.Unlock()
	}()
	guardMu.Lock()
	guard, err := attachChild(cmd)
	guardMu.Unlock()
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return fmt.Errorf("isolate image process: %w", err)
	}
	defer func() { guardMu.Lock(); _ = guard.Close(); guardMu.Unlock() }()
	err = cmd.Wait()
	log.flush()
	if err != nil {
		return fmt.Errorf("image runner failed: %w: %s", err, log.tail())
	}
	return nil
}

func buildArguments(p Profile, r Request, jobDir, output string, seed int64) []string {
	return buildArgumentsForMode(p, r, jobDir, output, seed, imageGPUResident)
}

func buildArgumentsForMode(p Profile, r Request, jobDir, output string, seed int64, mode imageMemoryMode) []string {
	args := []string{"--diffusion-model", p.DiffusionModel, "--llm", p.TextEncoder, "--vae", p.VAE,
		"--prompt-file", filepath.Join(jobDir, "prompt.txt"), "--negative-prompt-file", filepath.Join(jobDir, "negative.txt"),
		"--width", strconv.Itoa(p.Width), "--height", strconv.Itoa(p.Height), "--steps", strconv.Itoa(p.Steps),
		"--cfg-scale", strconv.FormatFloat(p.CFGScale, 'g', -1, 64), "--sampling-method", p.Sampler, "--seed", strconv.FormatInt(seed, 10), "--output", output}
	if p.Tokenizer != "" {
		args = append(args, "--tokenizer", p.Tokenizer)
	}
	if r.Inpaint != nil {
		args = append(args, "--init-img", r.ReferenceImages[0], "--mask", filepath.Join(jobDir, "mask.png"), "--strength", "1")
		if !r.crop.Empty() {
			args = append(args, "--inpaint-crop", fmt.Sprintf("%d,%d,%d,%d", r.crop.Min.X, r.crop.Min.Y, r.crop.Dx(), r.crop.Dy()))
		}
	}
	stage := "off"
	if p.StageComponents {
		stage = "on"
	}
	args = append(args, "--stage-components", stage, "--max-ram", strconv.FormatFloat(float64(p.MaxRAMMiB)/1024, 'f', -1, 64),
		"--vram-reserve", strconv.FormatFloat(float64(p.VRAMReserveMiB)/1024, 'f', -1, 64))
	backend := "cuda0"
	if p.VAEOnCPU {
		backend += ",vae=cpu"
	}
	if p.EncoderOnCPU {
		backend += ",te=cpu"
	}
	if mode == imageRAMCPU {
		backend = "cpu"
	}
	if mode != imageGPUResident {
		args = append(args, "--params-backend", "cpu", "--offload-to-cpu", "--auto-fit", "off")
	} else if p.OffloadToCPU {
		placement := backend
		if p.StageComponents {
			// Keep the denoiser resident across steps and free inactive GPU
			// copies of the encoder and VAE between phases.
			placement = "cuda0"
			if p.EncoderOnCPU {
				placement += ",te=cpu"
			} else {
				placement += ",te=disk"
			}
			if p.VAEOnCPU {
				placement += ",vae=cpu"
			} else {
				placement += ",vae=disk"
			}
		}
		args = append(args, "--params-backend", placement, "--auto-fit", "off")
		if !allowMiniMaxSegmentation(p, mode) {
			args = append(args, "--disable-segmented-compute")
		} else {
			args = append(args, miniMaxAttentionArguments(p, r)...)
		}
	} else if p.StageComponents {
		args = append(args, "--params-backend", "disk", "--auto-fit", "off")
	} else {
		args = append(args, "--auto-fit", "on")
	}
	args = append(args, "--backend", backend)
	if p.Sigmas != "" {
		args = append(args, "--sigmas", p.Sigmas)
	}
	if p.Scheduler != "" {
		args = append(args, "--scheduler", p.Scheduler)
	}
	if p.MaxVRAMMiB > 0 {
		args = append(args, "--max-vram", strconv.FormatFloat(float64(p.MaxVRAMMiB)/1024, 'f', -1, 64))
	}
	if p.CPUThreads > 0 {
		args = append(args, "--threads", strconv.Itoa(p.CPUThreads))
	}
	if p.FlashAttention {
		args = append(args, "--flash-attn", "on")
	} else {
		args = append(args, "--flash-attn", "off")
	}
	if p.SageAttention && mode != imageRAMCPU {
		args = append(args, "--sage-attn")
	}
	if p.VAETiling {
		args = append(args, "--vae-tiling")
	}
	if len(r.ReferenceImages) > 0 {
		if p.UsesMOSSEdit() {
			args = append(args, "--ref-image-args", "preset=moss_edit,vae_input_max_edge=384")
		} else if !sameComponentPath(p.VisionEncoder, p.TextEncoder) {
			args = append(args, "--llm_vision", p.VisionEncoder)
		}
		for _, path := range r.ReferenceImages {
			args = append(args, "-r", path)
		}
	}
	for _, lora := range p.requestLoRAs(r.ReferenceImages) {
		if lora.Scale != 0 {
			args = append(args, "--lora-model-dir", filepath.Join(jobDir, "loras"), "--lora-apply-mode", "at_runtime")
			break
		}
	}
	return args
}

func sameComponentPath(a, b string) bool {
	if runtime.GOOS == "windows" {
		return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
	}
	return filepath.Clean(a) == filepath.Clean(b)
}

func prepareLoRAs(ctx context.Context, loras []LoRA, jobDir string) (string, error) {
	var tags strings.Builder
	for i, lora := range loras {
		if lora.Scale == 0 {
			continue
		}
		if err := ctx.Err(); err != nil {
			return "", err
		}
		directory := filepath.Join(jobDir, "loras")
		if err := os.MkdirAll(directory, 0700); err != nil {
			return "", err
		}
		name := fmt.Sprintf("lora-%d%s", i, strings.ToLower(filepath.Ext(lora.Path)))
		target := filepath.Join(directory, name)
		// Same-volume links avoid duplicating selected local adapters. A bounded
		// streaming copy handles cross-volume profiles; it is job-local and never
		// changes the original LoRA or quantized base model.
		if err := os.Link(lora.Path, target); err != nil {
			source, err := os.Open(lora.Path)
			if err != nil {
				return "", err
			}
			output, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
			if err != nil {
				source.Close()
				return "", err
			}
			buffer := make([]byte, 1<<20)
			for {
				if err := ctx.Err(); err != nil {
					source.Close()
					output.Close()
					return "", err
				}
				n, readErr := source.Read(buffer)
				if n > 0 {
					if _, err := output.Write(buffer[:n]); err != nil {
						source.Close()
						output.Close()
						return "", err
					}
				}
				if readErr == io.EOF {
					break
				}
				if readErr != nil {
					source.Close()
					output.Close()
					return "", readErr
				}
			}
			source.Close()
			if err := output.Close(); err != nil {
				return "", err
			}
		}
		fmt.Fprintf(&tags, "\n<lora:%s:%s>", name, strconv.FormatFloat(lora.Scale, 'g', -1, 64))
	}
	return tags.String(), nil
}

func isolatedEnvironment(config Config, jobDir string) []string {
	env := os.Environ()
	set := func(key, value string) {
		prefix := strings.ToUpper(key) + "="
		for i := len(env) - 1; i >= 0; i-- {
			if strings.HasPrefix(strings.ToUpper(env[i]), prefix) {
				env = append(env[:i], env[i+1:]...)
			}
		}
		if value != "" {
			env = append(env, key+"="+value)
		}
	}
	for _, key := range []string{"TEMP", "TMP", "TMPDIR"} {
		set(key, jobDir)
	}
	for key, sub := range map[string]string{"CUDA_CACHE_PATH": "cuda-cache", "XDG_CACHE_HOME": "cache", "HF_HOME": "hf", "HF_HUB_CACHE": "hf/hub", "HF_XET_CACHE": "hf/xet", "LOCALAPPDATA": "local", "APPDATA": "appdata", "HOME": "home", "USERPROFILE": "home"} {
		set(key, filepath.Join(config.WorkDir, sub))
	}
	// This upstream variable names an optional plugin FILE, not a directory.
	// Clear a parent text-runner override; the image loader discovers its own
	// modules beside sd-cli and dependent DLLs resolve from that private folder.
	set("GGML_BACKEND_PATH", "")
	// Long GPU image kernels need no busy-waiting CPU core. This only affects
	// the isolated image process; the text runner and numerical work are intact.
	// Keep an explicit override for driver/platform latency investigations.
	if runtime.GOOS == "windows" && strings.TrimSpace(os.Getenv("WILDIMAGE_CUDA_WAIT")) == "" {
		set("WILDIMAGE_CUDA_WAIT", "blocking")
	}
	for _, key := range []string{"WILDIMAGE_CONDITION_CACHE_DIR", "WILDIMAGE_CONDITION_NAMESPACE", "WILDIMAGE_CONDITION_CACHE_BYTES"} {
		set(key, "")
	}
	set("PATH", filepath.Dir(config.RunnerPath)+string(os.PathListSeparator)+os.Getenv("PATH"))
	return env
}

var stepPattern = regexp.MustCompile(`(?:^|[^0-9])([0-9]{1,9})/([0-9]{1,9})(?:[^0-9]|$)`)
var byteRatePattern = regexp.MustCompile(`(?i)(?:^|[^a-z])(?:[kmgt]i?)?b/s\b`)

type progressLog struct {
	mu             sync.Mutex
	pending        string
	lines          []string
	callback       func(Progress)
	start          time.Time
	pid            int
	peakWorkingSet func() uint64
	peakRSSBytes   uint64
	lastPhase      string
	lastCurrent    int
	lastTotal      int
	lastEmit       time.Time
	counterPhase   string
	loadPhase      string
	expectedSteps  int
}

func (p *progressLog) Write(data []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pending += string(data)
	for {
		i := strings.IndexAny(p.pending, "\r\n")
		width := 1
		// Native bars start with CR and end with erase-to-EOL, without a
		// trailing newline until the final step. Consume the completed bar
		// now, not when the next step (or unrelated log) finally arrives.
		if end := strings.Index(p.pending, "\x1b[K"); end >= 0 && (i < 0 || end < i) {
			i, width = end, 3
		}
		if i < 0 {
			break
		}
		line := p.pending[:i]
		p.pending = p.pending[i+width:]
		p.line(line)
	}
	if len(p.pending) > 65536 {
		p.line(p.pending[:65536])
		p.pending = ""
	}
	return len(data), nil
}
func (p *progressLog) line(line string) {
	line = strings.TrimSpace(line)
	if line == "" {
		return
	}
	if len(line) > 1024 {
		line = line[:1024]
	}
	p.lines = append(p.lines, line)
	if len(p.lines) > 16 {
		p.lines = p.lines[len(p.lines)-16:]
	}
	lower := strings.ToLower(line)
	imageBatchHeader := strings.Contains(lower, "generating image:")
	samplerSelected := strings.Contains(lower, "sampling using ") || strings.Contains(lower, "sampling(high noise) using ")
	update := Progress{}
	switch {
	case strings.Contains(lower, "conditioning disk cache hit"):
		update.Phase = "conditioning_cache_hit"
		update.Message = "Reusing cached prompt encoding"
	case imageBatchHeader:
		update.Phase = "sampling"
		update.Message = "Generating image"
		update.Total = p.expectedSteps
	case samplerSelected:
		// The native request planner prints the sampler before reference
		// encoding. This is not the first diffusion step.
		update.Phase = "preparing"
		update.Message = "Preparing the image sampler"
		p.counterPhase = "preparing"
	case strings.Contains(lower, "loading tensors completed"):
		// Loading is finished, but the next (possibly untiled) GPU graph can
		// run silently for seconds. Resume its real phase immediately.
		update.Phase = p.counterPhase
		if update.Phase == "" {
			update.Phase = "preparing"
		}
	case strings.Contains(lower, "unloading") || strings.Contains(lower, "released inactive weights"):
		update.Phase = "releasing_components"
	case strings.Contains(lower, "loading") || strings.Contains(lower, "load tensors"):
		update.Phase = "loading"
		update.Message = "Loading image model components"
		switch {
		case strings.Contains(lower, "vae") || p.counterPhase == "decoding":
			update.Phase = "loading_vae"
		case strings.Contains(lower, "diffusion"):
			update.Phase = "loading_diffusion"
		case strings.Contains(lower, "text_encoder") || strings.Contains(lower, "loading llm") || strings.Contains(lower, "loading t5"):
			update.Phase = "loading_text_encoder"
		}
		p.loadPhase = update.Phase
	case strings.Contains(lower, "encoding") || strings.Contains(lower, "condition") || strings.Contains(lower, "vae encode") || strings.Contains(lower, "encode_first_stage"):
		update.Phase = "encoding"
		update.Message = "Encoding the prompt and reference images"
	case strings.Contains(lower, "decoding") || strings.Contains(lower, "vae decode") || strings.Contains(lower, "decode_first_stage"):
		update.Phase = "decoding"
		update.Message = "Decoding the image"
	case strings.Contains(lower, "saving") || strings.Contains(lower, "images saved"):
		update.Phase = "saving"
		update.Message = "Saving the image"
	}
	switch update.Phase {
	case "sampling", "encoding", "decoding", "saving":
		p.counterPhase = update.Phase
	}
	if match := stepPattern.FindStringSubmatch(line); match != nil && !imageBatchHeader {
		current, _ := strconv.Atoi(match[1])
		total, _ := strconv.Atoi(match[2])
		if total > 0 && current <= total {
			// Native tensor transfers and denoising both print unlabelled N/M
			// bars. Byte rates identify transfers; VAE tiles retain their semantic
			// phase even when loading weights interrupted the preceding message.
			if update.Phase == "" {
				if byteRatePattern.MatchString(line) {
					update.Phase = p.loadPhase
					if update.Phase == "" {
						update.Phase = "loading"
					}
					update.Message = "Loading image model components"
				} else if p.counterPhase == "decoding" {
					update.Phase = "decoding"
					update.Message = "Decoding the image"
				} else if p.counterPhase == "encoding" {
					update.Phase = "encoding"
					update.Message = "Encoding the prompt and reference images"
				} else if p.counterPhase == "saving" {
					update.Phase = "saving"
					update.Message = "Saving the image"
				} else {
					update.Phase = "sampling"
					update.Message = "Generating image"
				}
			}
			update.Current, update.Total = current, total
			if strings.Contains(lower, "tensors from") {
				// "loading 128/238 tensors from ..." selects 128 tensors out
				// of the file's 238; none has been transferred yet.
				update.Current, update.Total = 0, current
			}
		}
	}
	if update.Phase != "" {
		p.emit(update, false)
	}
}
func (p *progressLog) emit(v Progress, force bool) {
	if p.callback == nil {
		return
	}
	if !force && v.Phase == p.lastPhase && v.Current == p.lastCurrent && v.Total == p.lastTotal && time.Since(p.lastEmit) < 500*time.Millisecond {
		return
	}
	p.lastPhase = v.Phase
	p.lastCurrent = v.Current
	p.lastTotal = v.Total
	p.lastEmit = time.Now()
	v.PID = p.pid
	p.sampleWorkingSet()
	v.PeakRSSBytes = p.peakRSSBytes
	v.ElapsedSeconds = time.Since(p.start).Seconds()
	p.callback(v)
}

func (p *progressLog) sampleWorkingSet() {
	if p.peakWorkingSet != nil {
		p.peakRSSBytes = max(p.peakRSSBytes, p.peakWorkingSet())
	}
}
func (p *progressLog) flush() { p.mu.Lock(); defer p.mu.Unlock(); p.line(p.pending); p.pending = "" }
func (p *progressLog) tail() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return strings.Join(p.lines, "\n")
}
