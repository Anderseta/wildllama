package imagegen

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

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
