package imagegen

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestWanVideoPosterPreservesExactMiddlePacket(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "wan-five-frames.webm"))
	require.NoError(t, err)
	// Give each same-sized VP8 keyframe a distinct byte and retain its native packet.
	marker := []byte{0x9d, 0x01, 0x2a}
	var offsets []int
	for at := 0; at < len(data); {
		i := bytes.Index(data[at:], marker)
		if i < 0 {
			break
		}
		start := at + i - 3
		offsets = append(offsets, start)
		data[start+12] ^= byte(len(offsets))
		at = start + 13
	}
	require.Len(t, offsets, 5)
	path := filepath.Join(t.TempDir(), "video.webm")
	require.NoError(t, os.WriteFile(path, data, 0600))
	f, err := os.Open(path)
	require.NoError(t, err)
	defer f.Close()
	poster, err := MiddleVideoWebP(f, len(offsets))
	require.NoError(t, err)
	require.Equal(t, "WEBPVP8 ", string(poster[8:16]))
	size := int(binary.LittleEndian.Uint32(poster[16:20]))
	require.Equal(t, data[offsets[2]:offsets[2]+size], poster[20:20+size])
}

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
	t.Setenv("WILDLLAMA_WAN_ATTENTION", "inherited-value-must-be-ignored")
	for _, tc := range []struct {
		name      string
		attention string
		allow     bool
		backends  []string
	}{
		{"default", "", false, []string{"cuda0", "cuda0"}},
		{"strict", WanAttentionSageFP8, true, []string{"cuda0", "cuda0"}},
		{"opt-in", "", true, []string{"cuda0", "cuda0", "cpu"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			log := filepath.Join(dir, tc.name+".jsonl")
			t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", log)
			r := Request{Prompt: "literal --prompt ; &", OutputPath: filepath.Join(config.OutputDir, tc.name+".webm"), Video: &VideoOptions{Frames: 5, FPS: 16, AllowCPUFallback: tc.allow}}
			r.Video.AttentionBackend = tc.attention
			t.Setenv("WILDIMAGE_FAKE_EXPECT_WAN_ATTENTION", r.Video.AttentionBackend)
			_, err := engine.Run(t.Context(), p, r, nil)
			if tc.attention == WanAttentionSageFP8 {
				require.ErrorContains(t, err, "SageAttention")
			}
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

func TestWanVideoRetriesUnsupportedAttentionOnCUDA(t *testing.T) {
	dir := t.TempDir()
	exe, err := os.Executable()
	require.NoError(t, err)
	config := Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: filepath.Join(dir, "outputs")}
	engine, err := NewEngine(config)
	require.NoError(t, err)
	defer engine.Close()
	p := testProfile(t, dir)
	p.ModelID, p.Sampler, p.Scheduler, p.Sigmas = WanRapidT2V, "euler_a", "beta", ""
	p.OffloadToCPU = false // Attention recovery is independent of memory retries.
	fixture, err := filepath.Abs(filepath.Join("testdata", "wan-five-frames.webm"))
	require.NoError(t, err)
	attemptFile := filepath.Join(dir, "attempts.jsonl")
	t.Setenv("WILDIMAGE_FAKE_CHILD", "wan-attention-fallback")
	t.Setenv("WILDIMAGE_FAKE_WEBM", fixture)
	t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", attemptFile)
	video := &VideoOptions{Frames: 5, FPS: 16, AttentionBackend: WanAttentionSageFP8}
	r := Request{Prompt: "literal --prompt ; &", Seed: 1234, OutputPath: filepath.Join(config.OutputDir, "clip.webm"), Video: video}
	var notice string
	result, err := engine.Run(t.Context(), p, r, func(progress Progress) {
		if progress.Phase == "attention_fallback" {
			notice = progress.Message
		}
	})
	require.NoError(t, err)
	require.Contains(t, notice, "CuTe")
	require.Equal(t, WanAttentionSageFP8, video.AttentionBackend)
	require.Equal(t, WanAttentionCuTe, result.VideoAttentionUsed)
	require.NoError(t, checkWebM(result.Path, 5, 64, 64))
	attempts := imageAttempts(t, attemptFile)
	require.Len(t, attempts, 2)
	require.Equal(t, attempts[0], attempts[1], "only the attention environment should change")
	require.Equal(t, "1234", imageArgument(t, attempts[1], "--seed"))
}

func TestWanVideoAttentionTracksExecution(t *testing.T) {
	var w wanProgress
	w.observe("selected sage_fp8", 4)
	require.Empty(t, w.attentionUsed)
	_, publish := w.observe("wildvideo attention sage_fp8", 4)
	require.False(t, publish)
	require.Equal(t, WanAttentionSageFP8, w.attentionUsed)
	w.restart(1)
	require.Empty(t, w.attentionUsed)
	w.observe("wildvideo attention cute", 4)
	require.Equal(t, WanAttentionCuTe, w.attentionUsed)
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
	restored, ok := w.observe("wan restored 64.00 mib of weights from ram", 4)
	require.True(t, ok)
	require.True(t, restored.VideoHostWeights)
	require.Equal(t, "sampling", restored.Phase)
	require.Greater(t, last, 45)
	w.restart(1)
	p, ok := w.observe("img2vid", 4)
	require.True(t, ok)
	require.Equal(t, 2, p.VideoAttempt)
	require.Equal(t, last, p.VideoPercent)
	for _, event := range []string{"wildvideo reference reused", "generate_video 480x832x221", "wildvideo segment 122/122", "|### | 1/4 - 259.66s/it", "wildvideo segment 1/122"} {
		p, ok = w.observe(event, 4)
		require.True(t, ok, event)
		require.Equal(t, last, p.VideoPercent, event)
	}
	p, ok = w.observe("wildvideo segment 122/122", 4)
	require.True(t, ok)
	require.Equal(t, 58, p.VideoPercent)
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

func TestWanRestoresOnlyPublishedSamplerSteps(t *testing.T) {
	var w wanProgress
	w.restart(1)
	_, emitted := w.observe("wildvideo saved step 1/4 in ram", 4)
	require.False(t, emitted)
	require.Equal(t, 1, w.savedStep)
	w.restart(1)
	p, ok := w.observe("wildvideo resumed step 1/4", 4)
	require.True(t, ok)
	require.Equal(t, 1, p.Current)
	require.Equal(t, 44, p.VideoPercent)
	p, ok = w.observe("wildvideo segment 61/122", 4)
	require.True(t, ok)
	require.Equal(t, 51, p.VideoPercent)
	require.Contains(t, p.Message, "passo 2/4")
	_, _ = w.observe("wildvideo saved step 3/12 in ram", 4)
	require.Equal(t, 1, w.savedStep)
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
