package imagegen

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
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

	_ "golang.org/x/image/webp"
)

type Engine struct {
	config           Config
	mu               sync.Mutex
	closed           bool
	pid              int
	cancel           context.CancelFunc
	done             chan struct{}
	runtimeSize      int64
	runtimeModified  int64
	manifestModified int64
	runtimeRevision  string
}

// NewEngine is read-only: status/configuration requests never create a cache,
// start a process, or download weights. Write directories are created by Run.
func NewEngine(config Config) (*Engine, error) {
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
	if manifestInfo, err := os.Stat(manifestPath); err == nil {
		var manifest struct {
			Revision string `json:"revision"`
			Files    []struct {
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
		if e.runtimeSize != info.Size() || e.runtimeModified != info.ModTime().UnixNano() || e.manifestModified != manifestInfo.ModTime().UnixNano() {
			f, err := os.Open(e.config.RunnerPath)
			if err != nil {
				status.Error = err.Error()
				return status
			}
			hash := sha256.New()
			_, err = io.Copy(hash, f)
			f.Close()
			if err != nil || !strings.EqualFold(hex.EncodeToString(hash.Sum(nil)), executableHash) {
				status.Error = "image runner SHA256 differs from its pinned manifest"
				return status
			}
			e.runtimeSize = info.Size()
			e.runtimeModified = info.ModTime().UnixNano()
			e.manifestModified = manifestInfo.ModTime().UnixNano()
			e.runtimeRevision = manifest.Revision
		}
		status.Revision = e.runtimeRevision
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
	if len(p.LoRAs) > 8 {
		return errors.New("at most eight image LoRAs are supported")
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
	if r.Width != 0 {
		p.Width = r.Width
	}
	if r.Height != 0 {
		p.Height = r.Height
	}
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
	if len(r.ReferenceImages) > 4 {
		return errors.New("at most four reference images are supported per job")
	}
	if len(r.ReferenceImages) > 0 {
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
	for _, lora := range p.LoRAs {
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
	}
	for _, path := range r.ReferenceImages {
		if !filepath.IsAbs(path) {
			return errors.New("reference image paths must be absolute")
		}
		f, err := os.Open(path)
		if err != nil {
			return fmt.Errorf("reference image: %w", err)
		}
		info, statErr := f.Stat()
		if statErr != nil || !info.Mode().IsRegular() || info.Size() > 48<<20 {
			f.Close()
			return errors.New("reference image must be a regular image of at most 48 MiB")
		}
		cfg, format, decodeErr := image.DecodeConfig(io.LimitReader(f, 48<<20))
		f.Close()
		if decodeErr != nil || (format != "png" && format != "jpeg" && format != "webp") || cfg.Width <= 0 || cfg.Height <= 0 || int64(cfg.Width)*int64(cfg.Height) > 64<<20 {
			return errors.New("reference image must be a valid PNG, JPEG or WebP with at most 64 megapixels")
		}
	}
	return nil
}

func validateModelFile(path string) error {
	if path == "" || !filepath.IsAbs(path) {
		return errors.New("model component path must be absolute")
	}
	ext := strings.ToLower(filepath.Ext(path))
	if ext != ".gguf" && ext != ".safetensors" {
		return errors.New("expected a GGUF or safetensors component")
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() < 16 {
		return errors.New("model component is missing or truncated")
	}
	header := make([]byte, 8)
	if _, err := io.ReadFull(f, header); err != nil {
		return err
	}
	if ext == ".gguf" && string(header[:4]) != "GGUF" {
		return errors.New("invalid GGUF header")
	}
	if ext == ".safetensors" {
		n := binary.LittleEndian.Uint64(header)
		if n < 2 || n > 256<<20 || n > uint64(info.Size()-8) {
			return errors.New("invalid safetensors header length")
		}
	}
	return nil
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
	runnerStatus := e.Status()
	if !runnerStatus.Ready {
		return Result{}, errors.New(runnerStatus.Error)
	}
	if err := ValidateRequest(p, r); err != nil {
		return Result{}, err
	}
	if r.Width != 0 {
		p.Width = r.Width
	}
	if r.Height != 0 {
		p.Height = r.Height
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
	ctx, cancel := context.WithCancel(ctx)
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
	loraTags, err := prepareLoRAs(ctx, p.LoRAs, jobDir)
	if err != nil {
		return Result{}, err
	}
	if err := os.WriteFile(filepath.Join(jobDir, "prompt.txt"), []byte(r.Prompt+loraTags), 0600); err != nil {
		return Result{}, err
	}
	if err := os.WriteFile(filepath.Join(jobDir, "negative.txt"), []byte(r.NegativePrompt), 0600); err != nil {
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
	args := buildArguments(p, r, jobDir, output, seed)
	cmd := exec.CommandContext(ctx, e.config.RunnerPath, args...)
	cmd.Dir = jobDir
	cmd.Env = isolatedEnvironment(e.config, jobDir)
	prepareCommand(cmd)
	start := time.Now()
	log := &progressLog{callback: progress, start: start}
	cmd.Stdout = log
	cmd.Stderr = log
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
	log.emit(Progress{Phase: "starting", Message: "Starting the isolated CUDA image runner"}, true)
	if err := cmd.Start(); err != nil {
		return Result{}, fmt.Errorf("start image runner: %w", err)
	}
	e.mu.Lock()
	e.pid = cmd.Process.Pid
	e.mu.Unlock()
	log.mu.Lock()
	log.pid = cmd.Process.Pid
	log.mu.Unlock()
	guardMu.Lock()
	guard, err = attachChild(cmd)
	guardMu.Unlock()
	if err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return Result{}, fmt.Errorf("isolate image process: %w", err)
	}
	defer func() { guardMu.Lock(); _ = guard.Close(); guardMu.Unlock() }()
	err = cmd.Wait()
	log.flush()
	if ctx.Err() != nil {
		_ = os.Remove(output)
		return Result{}, ctx.Err()
	}
	if err != nil {
		_ = os.Remove(output)
		return Result{}, fmt.Errorf("image runner failed: %w: %s", err, log.tail())
	}
	f, err := os.Open(output)
	if err != nil {
		return Result{}, fmt.Errorf("runner did not produce an image: %w", err)
	}
	cfg, format, err := image.DecodeConfig(bufio.NewReader(f))
	f.Close()
	if err != nil || format != "png" || cfg.Width != p.Width || cfg.Height != p.Height {
		_ = os.Remove(output)
		return Result{}, errors.New("runner output is not a valid PNG with the requested dimensions")
	}
	result := Result{Path: output, MIMEType: "image/png", Width: cfg.Width, Height: cfg.Height, Seed: seed, DurationSeconds: time.Since(start).Seconds(), RunnerRevision: runnerStatus.Revision}
	log.emit(Progress{Phase: "completed", Message: "Image saved", Current: p.Steps, Total: p.Steps}, true)
	return result, nil
}

func buildArguments(p Profile, r Request, jobDir, output string, seed int64) []string {
	args := []string{"--diffusion-model", p.DiffusionModel, "--llm", p.TextEncoder, "--vae", p.VAE,
		"--prompt-file", filepath.Join(jobDir, "prompt.txt"), "--negative-prompt-file", filepath.Join(jobDir, "negative.txt"),
		"--width", strconv.Itoa(p.Width), "--height", strconv.Itoa(p.Height), "--steps", strconv.Itoa(p.Steps),
		"--cfg-scale", strconv.FormatFloat(p.CFGScale, 'g', -1, 64), "--sampling-method", p.Sampler, "--seed", strconv.FormatInt(seed, 10), "--output", output}
	if p.StageComponents {
		args = append(args, "--params-backend", "disk", "--auto-fit", "off")
	} else {
		args = append(args, "--auto-fit", "on")
	}
	backend := "cuda0"
	if p.VAEOnCPU {
		backend += ",vae=cpu"
	}
	if p.EncoderOnCPU {
		backend += ",te=cpu"
	}
	args = append(args, "--backend", backend)
	if p.Sigmas != "" {
		args = append(args, "--sigmas", p.Sigmas)
	}
	if p.Scheduler != "" {
		args = append(args, "--scheduler", p.Scheduler)
	}
	if p.MaxVRAMMiB > 0 {
		args = append(args, "--max-vram", strconv.FormatFloat(float64(p.MaxVRAMMiB)/1024, 'f', 4, 64))
	}
	if p.CPUThreads > 0 {
		args = append(args, "--threads", strconv.Itoa(p.CPUThreads))
	}
	if p.OffloadToCPU && !p.StageComponents {
		args = append(args, "--offload-to-cpu", "--mmap")
	}
	if p.FlashAttention {
		args = append(args, "--flash-attn", "on")
	} else {
		args = append(args, "--flash-attn", "off")
	}
	if p.VAETiling {
		args = append(args, "--vae-tiling")
	}
	if len(r.ReferenceImages) > 0 {
		if !sameComponentPath(p.VisionEncoder, p.TextEncoder) {
			args = append(args, "--llm_vision", p.VisionEncoder)
		}
		for _, path := range r.ReferenceImages {
			args = append(args, "-r", path)
		}
	}
	for _, lora := range p.LoRAs {
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
	set("PATH", filepath.Dir(config.RunnerPath)+string(os.PathListSeparator)+os.Getenv("PATH"))
	return env
}

var stepPattern = regexp.MustCompile(`(?:^|[^0-9])([0-9]{1,3})/([0-9]{1,3})(?:[^0-9]|$)`)
var byteRatePattern = regexp.MustCompile(`(?i)(?:^|[^a-z])(?:[kmgt]i?)?b/s\b`)

type progressLog struct {
	mu          sync.Mutex
	pending     string
	lines       []string
	callback    func(Progress)
	start       time.Time
	pid         int
	lastPhase   string
	lastCurrent int
	lastEmit    time.Time
}

func (p *progressLog) Write(data []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pending += string(data)
	for {
		i := strings.IndexAny(p.pending, "\r\n")
		if i < 0 {
			break
		}
		line := p.pending[:i]
		p.pending = p.pending[i+1:]
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
	update := Progress{}
	switch {
	case strings.Contains(lower, "loading") || strings.Contains(lower, "load tensors"):
		update.Phase = "loading"
		update.Message = "Loading image model components"
	case strings.Contains(lower, "encoding") || strings.Contains(lower, "condition"):
		update.Phase = "encoding"
		update.Message = "Encoding the prompt and reference images"
	case strings.Contains(lower, "decoding") || strings.Contains(lower, "vae decode"):
		update.Phase = "decoding"
		update.Message = "Decoding the image"
	case strings.Contains(lower, "saving") || strings.Contains(lower, "images saved"):
		update.Phase = "saving"
		update.Message = "Saving the image"
	}
	if match := stepPattern.FindStringSubmatch(line); match != nil {
		current, _ := strconv.Atoi(match[1])
		total, _ := strconv.Atoi(match[2])
		if total > 1 && current <= total {
			// Native tensor transfers and denoising both print unlabelled N/M
			// bars. Byte rates identify transfers; explicit phase labels win.
			if update.Phase == "" {
				if byteRatePattern.MatchString(line) {
					update.Phase = "loading"
					update.Message = "Loading image model components"
				} else {
					update.Phase = "sampling"
					update.Message = "Generating image"
				}
			}
			update.Current, update.Total = current, total
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
	if !force && v.Phase == p.lastPhase && v.Current == p.lastCurrent && time.Since(p.lastEmit) < 500*time.Millisecond {
		return
	}
	p.lastPhase = v.Phase
	p.lastCurrent = v.Current
	p.lastEmit = time.Now()
	v.PID = p.pid
	v.ElapsedSeconds = time.Since(p.start).Seconds()
	p.callback(v)
}
func (p *progressLog) flush() { p.mu.Lock(); defer p.mu.Unlock(); p.line(p.pending); p.pending = "" }
func (p *progressLog) tail() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return strings.Join(p.lines, "\n")
}
