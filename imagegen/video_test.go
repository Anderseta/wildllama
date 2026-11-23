package imagegen

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestWanVideoLateFailurePreservesCompletedFile(t *testing.T) {
	dir := t.TempDir()
	exe, err := os.Executable()
	require.NoError(t, err)
	config := Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: filepath.Join(dir, "outputs")}
	engine, err := NewEngine(config)
	require.NoError(t, err)
	defer engine.Close()
	fixture, err := filepath.Abs(filepath.Join("testdata", "wan-five-frames.webm"))
	require.NoError(t, err)
	log := filepath.Join(dir, "attempts.jsonl")
	t.Setenv("WILDIMAGE_FAKE_CHILD", "saved-video-oom")
	t.Setenv("WILDIMAGE_FAKE_WEBM", fixture)
	t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", log)
	p := testProfile(t, dir)
	p.ModelID, p.Sampler, p.Scheduler, p.Sigmas = WanRapidT2V, "euler_a", "beta", ""
	r := Request{Prompt: "literal --prompt ; &", Seed: 42, OutputPath: filepath.Join(config.OutputDir, "clip.webm"), Video: &VideoOptions{Frames: 5, FPS: 16}}
	result, err := engine.Run(t.Context(), p, r, nil)
	require.Error(t, err)
	require.Equal(t, r.OutputPath, result.Path, "%v", err)
	require.NoError(t, checkWebM(result.Path, 5, 64, 64))
	require.Len(t, imageAttempts(t, log), 1)
	p.Width, p.Height, p.MaxRAMMiB, r.Video.Frames = 480, 832, 512, 221
	require.ErrorContains(t, validateImageInputs(p, r), "tensor RGB")
}

func TestWanVideoNativeProgress(t *testing.T) {
	var last Progress
	log := progressLog{video: true, start: time.Now(), expectedSteps: 7, callback: func(v Progress) { last = v }}
	for _, stage := range []struct{ native, phase string }{
		{"sampling using Euler A method", "preparing"},
		{"IMG2VID", "encoding"},
		{"|### | 2/7 - 3.5s/it", "encoding"},
		{"generate_video 480x832x221", "sampling"},
		{"|#### | 3/7 - 1.5s/it", "sampling"},
		{"generating latent video completed, taking 4.2s", "decoding"},
		{"|### | 2/5 - 1.0s/it", "decoding"},
		{"decode_first_stage completed, taking 0.31s", "saving"},
	} {
		log.line(stage.native)
		require.Equal(t, stage.phase, last.Phase, stage.native)
	}
}

func TestWanVideoCPUFallbackRequiresOptIn(t *testing.T) {
	dir := t.TempDir()
	exe, err := os.Executable()
	require.NoError(t, err)
	config := Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: filepath.Join(dir, "outputs")}
	engine, err := NewEngine(config)
	require.NoError(t, err)
	defer engine.Close()
	p := testProfile(t, dir)
	p.ModelID, p.Sampler, p.Scheduler, p.Sigmas = WanRapidT2V, "euler_a", "beta", ""
	t.Setenv("WILDIMAGE_FAKE_CHILD", "gpu-oom-then-cpu")
	for _, tc := range []struct {
		name     string
		allow    bool
		backends []string
	}{
		{"default", false, []string{"cuda0", "cuda0"}},
		{"opt-in", true, []string{"cuda0", "cuda0", "cpu"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			log := filepath.Join(dir, tc.name+".jsonl")
			t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", log)
			r := Request{Prompt: "literal --prompt ; &", OutputPath: filepath.Join(config.OutputDir, tc.name+".webm"), Video: &VideoOptions{Frames: 5, FPS: 16, AllowCPUFallback: tc.allow}}
			_, err := engine.Run(t.Context(), p, r, nil)
			if !tc.allow {
				require.ErrorContains(t, err, "VRAM")
			}
			attempts := imageAttempts(t, log)
			require.Len(t, attempts, len(tc.backends))
			for i, args := range attempts {
				require.Equal(t, tc.backends[i], imageArgument(t, args, "--backend"))
			}
		})
	}
}

func TestWanVideoProgressAcrossSegmentsAndRetry(t *testing.T) {
	var w wanProgress
	w.restart(1)
	events := []string{"img2vid", "get_clip_vision_output completed", "wildvideo tile 1/4",
		"wildvideo chunk 2/4", "wildvideo stage text", "generate_video 480x832x221",
		"wildvideo segment 0/42", "wildvideo segment 21/42", "|### | 1/4 - 6.5s/it",
		"wildvideo segment 0/42", "wildvideo segment 20/42"}
	last := 0
	for _, event := range events {
		p, ok := w.observe(event, 4)
		require.True(t, ok, event)
		require.GreaterOrEqual(t, p.VideoPercent, last, event)
		last = p.VideoPercent
	}
	_, ok := w.observe("|##### | 256/256 - 12.0mb/s", 4)
	require.False(t, ok)
	require.Greater(t, last, 45)
	w.restart(1)
	p, ok := w.observe("img2vid", 4)
	require.True(t, ok)
	require.Equal(t, 2, p.VideoAttempt)
	require.GreaterOrEqual(t, p.VideoPercent, last)
}

func TestWanVideoModesAndArguments(t *testing.T) {
	p := Profile{ModelID: WanRapidI2V, Sampler: "euler_a", Scheduler: "beta", DiffusionModel: "wan.gguf", TextEncoder: "umt5.gguf", VisionEncoder: "clip.safetensors", VAE: "vae.safetensors", StageComponents: true, OffloadToCPU: true, VAETiling: true}
	r := Request{Video: &VideoOptions{Frames: 33, FPS: 16}, ReferenceImages: []string{"first.png"}}
	require.NoError(t, validateVideoMode(p, r))
	args := buildArguments(p, r, "dir", "film.webm", 42)
	for _, pair := range [][]string{{"-M", "vid_gen"}, {"--init-img", "first.png"}, {"--clip_vision", p.VisionEncoder}, {"--t5xxl", p.TextEncoder}, {"--stage-components", "on"}, {"--auto-fit", "on"}, {"--video-frames", "33"}, {"--fps", "16"}, {"--sampling-method", "euler_a"}, {"--scheduler", "beta"}} {
		require.Contains(t, strings.Join(args, "\x00")+"\x00", pair[0]+"\x00"+pair[1]+"\x00")
	}
	require.NotContains(t, args, "--llm")
	p.ModelID = WanRapidT2V
	require.Error(t, validateVideoMode(p, r))
	r.ReferenceImages = nil
	require.NoError(t, validateVideoMode(p, r))
	args = buildArguments(p, r, "dir", "film.webm", 42)
	require.NotContains(t, args, "--clip_vision")
	require.NotContains(t, args, "--init-img")
	r.Video.Frames = 32
	require.Error(t, validateVideoMode(p, r))
}
func TestWanVideoOnlyPublishesCompleteOutput(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "wan-five-frames.webm"))
	require.NoError(t, err)
	dir := t.TempDir()
	part, target := filepath.Join(dir, "part.webm"), filepath.Join(dir, "video.webm")
	require.NoError(t, os.WriteFile(part, data[:len(data)/2], 0600))
	require.Error(t, publishWebM(part, target, 5, 64, 64))
	require.NoFileExists(t, target)
	require.NoError(t, os.WriteFile(part, data, 0600))
	require.Error(t, publishWebM(part, target, 9, 64, 64))
	require.ErrorContains(t, publishWebM(part, target, 5, 128, 64), "expected 128x64")
	require.NoError(t, publishWebM(part, target, 5, 64, 64))
	require.FileExists(t, target)
}
