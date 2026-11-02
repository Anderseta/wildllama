package imagegen

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fixtureTensor struct {
	name, dtype string
	shape       []int64
	data        []byte
}

func writeSafeFixture(t *testing.T, path string, items []fixtureTensor) {
	t.Helper()
	header := map[string]any{}
	payload := []byte{}
	for _, item := range items {
		start := len(payload)
		payload = append(payload, item.data...)
		header[item.name] = tensorDescriptor{DType: item.dtype, Shape: item.shape, Offsets: []int64{int64(start), int64(len(payload))}}
	}
	encoded, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, uint64(len(encoded)))
	data = append(data, encoded...)
	data = append(data, payload...)
	putModel(t, path, data)
}
func TestInspectPackedConvRotAndEmbeddedVision(t *testing.T) {
	path := filepath.Join(t.TempDir(), "encoder.safetensors")
	for _, group := range []int{256, 3} {
		metadata, _ := json.Marshal(map[string]any{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": group})
		writeSafeFixture(t, path, []fixtureTensor{
			{"model.layers.0.mlp.down_proj.weight", "I8", []int64{2, 256}, make([]byte, 512)},
			{"model.layers.0.mlp.down_proj.weight_scale", "F32", []int64{2, 1}, make([]byte, 8)},
			{"model.layers.0.mlp.down_proj.comfy_quant", "U8", []int64{int64(len(metadata))}, metadata},
			{"model.visual.patch_embed.proj.weight", "BF16", []int64{2, 2}, make([]byte, 8)},
			{"model.visual.merger.linear_fc2.weight", "BF16", []int64{2, 2}, make([]byte, 8)},
		})
		inspection, err := InspectLocalComponent(path, "text_encoder")
		if err != nil {
			t.Fatal(err)
		}
		if group == 256 {
			if inspection.Compatibility != "supported" || !inspection.SupportsEditing || !strings.Contains(inspection.Format, "ConvRot") {
				t.Fatal(inspection)
			}
		} else if inspection.Compatibility == "supported" {
			t.Fatal("unsupported rotation accepted")
		}
	}
}
func TestRuntimeLoRAAndDiskStageArguments(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "selected.safetensors")
	writeSafeFixture(t, path, []fixtureTensor{
		{"transformer_blocks.0.attn.to_q.lora_A.default.weight", "BF16", []int64{2, 4}, make([]byte, 16)},
		{"transformer_blocks.0.attn.to_q.lora_B.default.weight", "BF16", []int64{4, 2}, make([]byte, 16)},
	})
	inspection, err := InspectLocalComponent(path, "lora")
	if err != nil || inspection.Compatibility != "supported" {
		t.Fatal(inspection, err)
	}
	before, _ := os.ReadFile(path)
	job := filepath.Join(dir, "job")
	os.MkdirAll(job, 0700)
	loras := []LoRA{{Path: path, Scale: 1}, {Path: filepath.Join(dir, "missing.safetensors"), Scale: 0}}
	tags, err := prepareLoRAs(context.Background(), loras, job)
	if err != nil {
		t.Fatal(err)
	}
	if tags != "\n<lora:lora-0.safetensors:1>" {
		t.Fatal("unsafe or incorrect LoRA tag", tags)
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatal("selected LoRA was modified")
	}
	p := DefaultProfile("local-base")
	p.Steps = 25
	p.Sigmas = ""
	p.Scheduler = "simple"
	p.LoRAs = loras
	p.TextEncoder = "encoder.safetensors"
	p.VisionEncoder = p.TextEncoder
	if err := ValidateProfile(p); err != nil {
		t.Fatal(err)
	}
	args := strings.Join(buildArguments(p, Request{ReferenceImages: []string{"input.png"}}, job, "output.png", 42), "|")
	for _, required := range []string{"--params-backend|disk", "--auto-fit|off", "--scheduler|simple", "--lora-apply-mode|at_runtime", "--lora-model-dir|", "-r|input.png"} {
		if !strings.Contains(args, required) {
			t.Fatalf("missing %s: %s", required, args)
		}
	}
	for _, forbidden := range []string{"--mmap", "--eager-load", "--llm_vision", "--offload-to-cpu", "--type", "--sigmas"} {
		if strings.Contains(args, forbidden) {
			t.Fatalf("unrequested flag %s", forbidden)
		}
	}
	p.StageComponents = false
	legacy := strings.Join(buildArguments(p, Request{}, job, "output.png", 42), "|")
	if !strings.Contains(legacy, "--mmap") || !strings.Contains(legacy, "--offload-to-cpu") {
		t.Fatal("legacy opt-in changed")
	}
}
