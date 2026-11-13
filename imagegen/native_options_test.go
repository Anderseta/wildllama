package imagegen

import (
	"strings"
	"testing"
)

func TestMiniMaxPlacementPreservesGPUWeightsAndFallbacks(t *testing.T) {
	names := []string{"model.diffusion_model.video_patch_proj.weight", "condition_proj.weight"}
	h3, _ := writeGGUFTensorFixture(t, names)
	video, _ := writeGGUFTensorFixture(t, append(names, "audio_patch_proj.weight"))
	qwen, _ := writeGGUFTensorFixture(t, []string{"txt_in.weight"})
	p := DefaultProfile(MiniMaxH3RecipeID)
	for _, tc := range []struct {
		name, path string
		stage      bool
		mode       imageMemoryMode
		segmented  bool
	}{
		{"image generation", h3, true, imageGPUResident, true},
		{"video unchanged", video, true, imageGPUResident, false},
		{"Qwen unchanged", qwen, true, imageGPUResident, false},
		{"unknown unchanged", "missing.gguf", true, imageGPUResident, false},
		{"staging disabled", h3, false, imageGPUResident, false},
		{"RAM fallback", h3, true, imageRAMGPU, false},
		{"CPU fallback", h3, true, imageRAMCPU, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p.DiffusionModel, p.StageComponents = tc.path, tc.stage
			if got := allowMiniMaxSegmentation(p, tc.mode); got != tc.segmented {
				t.Fatalf("segmentation %v, want %v", got, tc.segmented)
			}
			args := strings.Join(buildArgumentsForMode(p, Request{ReferenceImages: []string{"one.png", "two.png"}}, "job", "out.png", 42, tc.mode), "|")
			if tc.mode == imageGPUResident {
				if strings.Contains(args, "--disable-segmented-compute") == tc.segmented {
					t.Fatal(args)
				}
				if !strings.Contains(args, "--params-backend|cuda0") {
					t.Fatal(args)
				}
			} else if !strings.Contains(args, "--params-backend|cpu|--offload-to-cpu") {
				t.Fatal(args)
			}
			if strings.Contains(args, "minimax_h3_attn_chunk_budget=0.03125") != tc.segmented {
				t.Fatal(args)
			}
			for _, required := range []string{"--steps|6", "--cfg-scale|1", "--sigmas|" + MiniMaxH3Sigmas, "--vram-reserve|0.25"} {
				if !strings.Contains(args, required) {
					t.Fatal(args)
				}
			}
		})
	}
	p.DiffusionModel, p.OffloadToCPU, p.StageComponents = h3, false, true
	if allowMiniMaxSegmentation(p, imageGPUResident) {
		t.Fatal("offload preference ignored")
	}
}

func TestMiniMaxAttentionGroupingOnlyWhenUseful(t *testing.T) {
	p := DefaultProfile(MiniMaxH3RecipeID)
	for _, tc := range []struct {
		refs, memory int
		want         bool
	}{
		{0, 14500, false}, {1, 14500, false}, {2, 14500, true},
		{4, 16384, true}, {2, 24576, false},
	} {
		p.MaxVRAMMiB = tc.memory
		got := miniMaxAttentionArguments(p, Request{ReferenceImages: make([]string, tc.refs)})
		if (len(got) > 0) != tc.want {
			t.Fatalf("%+v: %v", tc, got)
		}
	}
}
