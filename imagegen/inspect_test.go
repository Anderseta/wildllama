package imagegen

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

type fixtureTensor struct {
	name, dtype string
	shape       []int64
	data        []byte
}

func writeSafeFixture(t *testing.T, path string, items []fixtureTensor, metadata ...map[string]string) {
	t.Helper()
	header := map[string]any{}
	if len(metadata) > 0 {
		header["__metadata__"] = metadata[0]
	}
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
func TestRuntimeLoRAAndRAMStageArguments(t *testing.T) {
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
	first := strings.Join(buildArguments(p, Request{}, job, "output.png", 42), "|")
	if !strings.Contains(first, "--params-backend|cuda0") || !strings.Contains(first, "--disable-segmented-compute") || strings.Contains(first, "--offload-to-cpu") {
		t.Fatal("RAM offload was forced before checking GPU fit", first)
	}
	args := strings.Join(buildArgumentsForMode(p, Request{ReferenceImages: []string{"input.png"}}, job, "output.png", 42, imageRAMGPU), "|")
	for _, required := range []string{"--params-backend|cpu", "--offload-to-cpu", "--auto-fit|off", "--stage-components|on", "--max-ram|0", "--scheduler|simple", "--lora-apply-mode|at_runtime", "--lora-model-dir|", "-r|input.png"} {
		if !strings.Contains(args, required) {
			t.Fatalf("missing %s: %s", required, args)
		}
	}
	for _, forbidden := range []string{"--mmap", "--eager-load", "--llm_vision", "--type", "--sigmas", "--disable-segmented-compute"} {
		if strings.Contains(args, forbidden) {
			t.Fatalf("unrequested flag %s", forbidden)
		}
	}
	p.StageComponents = false
	resident := strings.Join(buildArgumentsForMode(p, Request{}, job, "output.png", 42, imageRAMGPU), "|")
	if strings.Contains(resident, "--mmap") || !strings.Contains(resident, "--offload-to-cpu") || !strings.Contains(resident, "--stage-components|off") {
		t.Fatal("RAM offload was disabled with phase staging", resident)
	}
	p.StageComponents, p.OffloadToCPU = true, false
	disk := strings.Join(buildArguments(p, Request{}, job, "output.png", 42), "|")
	if !strings.Contains(disk, "--params-backend|disk") || strings.Contains(disk, "--offload-to-cpu") {
		t.Fatal("explicit disk-only mode changed", disk)
	}
}

func TestInspectionCacheInvalidationAndOwnership(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "encoder.safetensors")
	makeFixture := func(path, dtype string) {
		writeSafeFixture(t, path, []fixtureTensor{
			{"visual.patch_embed", dtype, []int64{1}, []byte{0, 0}},
			{"visual.merger.linear_fc2", dtype, []int64{1}, []byte{0, 0}},
		})
	}
	makeFixture(path, "F16")
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "windows" {
		time.Sleep(1100 * time.Millisecond)
	} // allow NTFS change stamps to settle
	first, err := InspectLocalComponent(path, "text_encoder")
	if err != nil || !first.SupportsEditing || first.DTypes["F16"] != 2 {
		t.Fatal(first, err)
	}
	first.DTypes["F16"] = 99
	second, err := InspectLocalComponent(path, "text_encoder")
	if err != nil || second.DTypes["F16"] != 2 {
		t.Fatal(second, err)
	}
	second.DTypes["F16"] = 55
	vae, err := InspectLocalComponent(path, "vae")
	if err != nil || vae.SupportsEditing {
		t.Fatal(vae, err)
	}
	makeFixture(path, "F32")
	if err := os.Chtimes(path, stat.ModTime(), stat.ModTime()); err != nil {
		t.Fatal(err)
	}
	got, err := InspectLocalComponent(path, "text_encoder")
	if err != nil || got.DTypes["F32"] != 2 || got.DTypes["F16"] != 0 {
		t.Fatal(got, err)
	}
	bad := filepath.Join(dir, "replacement.safetensors")
	putModel(t, bad, make([]byte, stat.Size()))
	if err := os.Chtimes(bad, stat.ModTime(), stat.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(bad, path); err != nil {
		t.Fatal(err)
	}
	if _, err := InspectLocalComponent(path, "text_encoder"); err == nil {
		t.Fatal("accepted invalid replacement")
	}
}
func TestImageRunnerStatusChecksPinnedDigestAfterChange(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "runner.exe")
	data := []byte("test runner")
	putModel(t, path, data)
	manifestPath := filepath.Join(dir, "wildimage-runtime.json")
	manifest := func(hash, version string) {
		b, err := json.Marshal(map[string]any{"revision": version, "features": map[string]bool{"ram_budget": true}, "files": []map[string]any{{"name": "runner.exe", "size": len(data), "sha256": hash}}})
		if err != nil {
			t.Fatal(err)
		}
		putModel(t, manifestPath, b)
	}
	manifest(digest(data), "v1")
	info, err := os.Stat(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	engine := &Engine{config: Config{RunnerPath: path}}
	if got := engine.Status(); !got.Ready || got.Revision != "v1" {
		t.Fatal(got)
	}
	manifest(strings.Repeat("f", 64), "v2")
	if err := os.Chtimes(manifestPath, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if got := engine.Status(); got.Ready || !strings.Contains(got.Error, "SHA256") {
		t.Fatal(got)
	}
	manifest(digest(data), "v2")
	if got := engine.Status(); !got.Ready || got.Revision != "v2" {
		t.Fatal(got)
	}
	binaryInfo, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	data[0] ^= 1
	putModel(t, path, data)
	if err := os.Chtimes(path, binaryInfo.ModTime(), binaryInfo.ModTime()); err != nil {
		t.Fatal(err)
	}
	if got := engine.Status(); got.Ready || !strings.Contains(got.Error, "SHA256") {
		t.Fatal(got)
	}
}
