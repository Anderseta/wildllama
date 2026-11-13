package imagegen

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMiniMaxRecipeAndEmbeddedVision(t *testing.T) {
	var recipe Recipe
	for _, r := range Recipes() {
		if r.ID == MiniMaxH3RecipeID {
			recipe = r
		}
	}
	p := recipe.DefaultProfile
	if err := ValidateProfile(p); err != nil {
		t.Fatal(err)
	}
	if p.Steps != 6 || p.CFGScale != 1 || p.Sigmas != MiniMaxH3Sigmas || p.SageAttention ||
		!p.StageComponents || !p.OffloadToCPU || !p.FlashAttention || p.VRAMReserveMiB != 256 {
		t.Fatalf("wrong H3 preset: %+v", p)
	}
	b := Bundle{ID: "local-h3", RecipeID: recipe.ID, Profile: p}
	for _, c := range recipe.Components {
		b.Components = append(b.Components, InstalledComponent{Component: c, Path: filepath.Join(t.TempDir(), filepath.Base(c.Filename))})
	}
	bindProfile(&b)
	if len(b.Components) != 3 || b.Profile.VisionEncoder != b.Profile.TextEncoder {
		t.Fatal("duplicated or missing embedded vision")
	}
	args := buildArgumentsForMode(b.Profile, Request{ReferenceImages: []string{"one.png", "two.png"}}, "job", "output.png", 7, imageGPUResident)
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "--llm_vision") || !strings.Contains(joined, "-r one.png -r two.png") || !strings.Contains(joined, MiniMaxH3Sigmas) {
		t.Fatal(joined)
	}
	old := DefaultProfile(DefaultRecipeID)
	if old.Steps != 8 || old.Sigmas != TurboSigmas {
		t.Fatal("Qwen default changed")
	}
}

func TestInspectGGUFCombinedEncoder(t *testing.T) {
	for _, vision := range []bool{false, true} {
		names := []string{"layers.0.weight", "layers.1.weight"}
		if vision {
			names = []string{"visual.patch_embed.proj.weight", "visual.merger.linear_fc2.weight"}
		}
		path, size := writeGGUFTensorFixture(t, names)
		inspection, err := InspectLocalComponent(path, "text_encoder")
		if err != nil || inspection.SupportsEditing != vision || inspection.TensorCount != 2 {
			t.Fatalf("%+v %v", inspection, err)
		}
		if err := os.Truncate(path, int64(size-1)); err != nil {
			t.Fatal(err)
		}
		if _, err := InspectLocalComponent(path, "text_encoder"); err == nil {
			t.Fatal("truncated GGUF accepted")
		}
	}
}

func writeGGUFTensorFixture(t *testing.T, names []string) (string, int) {
	t.Helper()
	var buf bytes.Buffer
	put := func(v any) {
		if err := binary.Write(&buf, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	buf.WriteString("GGUF")
	put(uint32(3))
	put(uint64(len(names)))
	put(uint64(0))
	for i, name := range names {
		put(uint64(len(name)))
		buf.WriteString(name)
		put(uint32(1))
		put(uint64(8))
		put(uint32(0))
		put(uint64(i * 32))
	}
	for buf.Len()%32 != 0 {
		buf.WriteByte(0)
	}
	buf.Write(make([]byte, len(names)*32))
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return path, buf.Len()
}
