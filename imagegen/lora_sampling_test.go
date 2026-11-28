package imagegen

import (
	"math"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
)

func TestOmniTurboLoRASampling(t *testing.T) {
	standard := DefaultProfile(OmniFP8RecipeID)
	standard.LoRAs = []LoRA{{Path: filepath.Join(t.TempDir(), "omni.safetensors"), Scale: 1, Preset: omniTurboPreset}}
	if err := ValidateProfile(standard); err != nil {
		t.Fatal(err)
	}
	active := standard.WithLoRASampling()
	if active.Steps != 8 || active.CFGScale != 1 || active.Sampler != "euler" || active.Scheduler != "" {
		t.Fatalf("unexpected sampling settings: %+v", active)
	}
	expected := []float64{1, 0.933358293, 0.857191977, 0.769301982, 0.666755818, 0.545554016, 0.400096293, 0.222291574, 0}
	sigmas := strings.Split(active.Sigmas, ",")
	if len(sigmas) != len(expected) {
		t.Fatalf("expected nine noise levels: %s", active.Sigmas)
	}
	for i, value := range sigmas {
		parsed, err := strconv.ParseFloat(value, 32)
		if err != nil || math.Abs(parsed-expected[i]) > 1e-7 {
			t.Fatalf("sigma %d: got %s, want %v", i, value, expected[i])
		}
	}
	// Native arguments must use the generation canvas even if sampling was
	// already resolved before an inpaint crop or resize.
	active.Width, active.Height = 768, 512
	args := buildArguments(active, Request{}, ".", "output.png", 42)
	i := slices.Index(args, "--sigmas")
	if i == -1 || args[i+1] == active.Sigmas || args[i+1] != omniTurboSigmas(768, 512, 8) {
		t.Fatalf("sigmas did not follow the final canvas: %v", args)
	}
	if got := omniTurboSigmas(512, 768, 8); got != args[i+1] {
		t.Fatalf("equivalent target token count changed the schedule: %s", got)
	}
	standard.LoRAs[0].Scale, standard.LoRAs[0].Preset = 0, ""
	restored := standard.WithLoRASampling()
	if restored.Sigmas != "" || restored.Steps != 35 || restored.CFGScale != 5 || restored.Sampler != "er_sde" {
		t.Fatalf("expected original sampling settings: %+v", restored)
	}
}

func TestOmniTurboLoRARejectsInvalidActivation(t *testing.T) {
	p := DefaultProfile(OmniFP8RecipeID)
	adapter := LoRA{Path: filepath.Join(t.TempDir(), "omni.safetensors"), Scale: 1, Preset: omniTurboPreset}
	p.LoRAs = []LoRA{adapter, adapter}
	if err := ValidateProfile(p); err == nil {
		t.Fatal("accepted two sampling presets")
	}
	p.LoRAs = []LoRA{adapter}
	p.LoRAs[0].Scale = 0
	if err := ValidateProfile(p); err == nil {
		t.Fatal("accepted an inactive LoRA with an active sampling preset")
	}
}
