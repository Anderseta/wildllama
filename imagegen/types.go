// Package imagegen runs an isolated stable-diffusion.cpp process. It never
// loads image ggml libraries into the text inference process.
package imagegen

import "context"

const (
	RunnerRevision  = "3be2378"
	RunnerRelease   = "master-1043-3be2378"
	DefaultRecipeID = "qwen-image-2.1-turbo-uncensored-q5"
	ManifestName    = "wildimage-model.json"
	TurboSigmas     = "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"
)

type Config struct {
	RunnerPath           string `json:"runner_path"`
	WorkDir              string `json:"work_dir"`
	ModelsDir            string `json:"models_dir"`
	OutputDir            string `json:"output_dir"`
	ConditioningCacheMiB int    `json:"conditioning_cache_mib"`
	MinFreeDiskMiB       int    `json:"min_free_disk_mib"`
}

type Profile struct {
	ID              string  `json:"id"`
	ModelID         string  `json:"model_id"`
	DiffusionModel  string  `json:"diffusion_model"`
	TextEncoder     string  `json:"text_encoder"`
	VAE             string  `json:"vae"`
	VisionEncoder   string  `json:"vision_encoder,omitempty"`
	Width           int     `json:"width"`
	Height          int     `json:"height"`
	Steps           int     `json:"steps"`
	CFGScale        float64 `json:"cfg_scale"`
	Sampler         string  `json:"sampler"`
	Sigmas          string  `json:"sigmas"`
	Scheduler       string  `json:"scheduler,omitempty"`
	LoRAs           []LoRA  `json:"loras,omitempty"`
	StageComponents bool    `json:"stage_components"`
	// PreserveReferenceSize prefers a supported canvas from the first reference.
	PreserveReferenceSize bool `json:"preserve_reference_size"`
	// MaxVRAMMiB is the runner's streaming/fitting target, not a process hard cap.
	MaxVRAMMiB     int  `json:"max_vram_mib"`
	CPUThreads     int  `json:"cpu_threads"`
	OffloadToCPU   bool `json:"offload_to_cpu"`
	FlashAttention bool `json:"flash_attention"`
	VAETiling      bool `json:"vae_tiling"`
	VAEOnCPU       bool `json:"vae_on_cpu"`
	EncoderOnCPU   bool `json:"encoder_on_cpu"`
}

type LoRA struct {
	Path  string  `json:"path"`
	Scale float64 `json:"scale"`
}
type LocalInspection struct {
	Path            string         `json:"path"`
	Role            string         `json:"role"`
	Format          string         `json:"format"`
	Size            int64          `json:"size"`
	TensorCount     int            `json:"tensor_count"`
	DTypes          map[string]int `json:"dtypes,omitempty"`
	Compatibility   string         `json:"compatibility"`
	Hint            string         `json:"hint"`
	SupportsEditing bool           `json:"supports_editing"`
}

type Request struct {
	Prompt          string   `json:"prompt"`
	NegativePrompt  string   `json:"negative_prompt,omitempty"`
	Width           int      `json:"width,omitempty"`
	Height          int      `json:"height,omitempty"`
	Seed            int64    `json:"seed"`
	ReferenceImages []string `json:"reference_images,omitempty"`
	OutputPath      string   `json:"output_path"`
}

type Progress struct {
	Phase          string  `json:"phase"`
	Message        string  `json:"message"`
	Current        int     `json:"current,omitempty"`
	Total          int     `json:"total,omitempty"`
	PID            int     `json:"pid,omitempty"`
	ElapsedSeconds float64 `json:"elapsed_seconds"`
}

type Result struct {
	Path            string  `json:"path"`
	MIMEType        string  `json:"mime_type"`
	Width           int     `json:"width"`
	Height          int     `json:"height"`
	Seed            int64   `json:"seed"`
	DurationSeconds float64 `json:"duration_seconds"`
	RunnerRevision  string  `json:"runner_revision"`
}

type RunnerStatus struct {
	Ready                      bool   `json:"ready"`
	Running                    bool   `json:"running"`
	PID                        int    `json:"pid,omitempty"`
	RunnerPath                 string `json:"runner_path"`
	Revision                   string `json:"revision"`
	Error                      string `json:"error,omitempty"`
	ConditioningCacheSupported bool   `json:"conditioning_cache_supported"`
}

type Component struct {
	Role     string `json:"role"`
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
	Filename string `json:"filename"`
	SHA256   string `json:"sha256"`
	Size     int64  `json:"size"`
	Required bool   `json:"required"`
}
type InstalledComponent struct {
	Component
	Path            string `json:"path"`
	Ready           bool   `json:"ready"`
	Error           string `json:"error,omitempty"`
	DownloadedBytes int64  `json:"downloaded_bytes"`
	DownloadState   string `json:"download_state"`
}
type Recipe struct {
	ID             string      `json:"id"`
	Name           string      `json:"name"`
	Family         string      `json:"family"`
	Capabilities   []string    `json:"capabilities"`
	Components     []Component `json:"components"`
	DefaultProfile Profile     `json:"default_profile"`
}
type Bundle struct {
	ID         string               `json:"id"`
	Name       string               `json:"name"`
	RecipeID   string               `json:"recipe_id,omitempty"`
	Directory  string               `json:"directory"`
	Components []InstalledComponent `json:"components"`
	Profile    Profile              `json:"profile"`
	Ready      bool                 `json:"ready"`
	Missing    []string             `json:"missing"`
}
type DownloadProgress struct {
	Phase     string `json:"phase"`
	Role      string `json:"role"`
	Filename  string `json:"filename"`
	Message   string `json:"message"`
	Completed int64  `json:"completed"`
	Total     int64  `json:"total"`
	FileIndex int    `json:"file_index"`
	FileCount int    `json:"file_count"`
}

// Runner lets the API test orchestration without loading model weights.
type Runner interface {
	Run(context.Context, Profile, Request, func(Progress)) (Result, error)
	Status() RunnerStatus
	Close() error
}

func DefaultProfile(modelID string) Profile {
	return Profile{ID: "qwen-image-2.1-turbo", ModelID: modelID, Width: 1024, Height: 1024,
		Steps: 8, CFGScale: 1, Sampler: "euler", Sigmas: TurboSigmas, MaxVRAMMiB: 12288,
		StageComponents: true, OffloadToCPU: true, FlashAttention: true, VAETiling: true}
}
