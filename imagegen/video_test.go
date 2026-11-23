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
