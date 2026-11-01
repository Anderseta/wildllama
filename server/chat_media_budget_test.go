package server

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/png"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func projectorModel(kind string, values map[string]int) *Model {
	kv := map[string]any{"general.architecture": "clip", "clip.vision.projector_type": kind}
	for key, value := range values {
		encoded, _ := json.Marshal(value)
		kv["clip.vision."+key] = json.Number(encoded)
	}
	return &Model{ProjectorPaths: []string{"projector.gguf"}, projectorMetadata: []ggufMetadata{{KV: kv}}}
}

func TestProjectorImageBudgetUsesDimensionsAndModel(t *testing.T) {
	qwen := projectorModel("qwen2vl", map[string]int{"patch_size": 14, "spatial_merge_size": 2})
	small, large := projectorImageTokens(qwen, 224, 224), projectorImageTokens(qwen, 2048, 2048)
	if small != 66 || large != 4098 {
		t.Fatalf("Qwen budgets small=%d large=%d", small, large)
	}
	gemma := projectorModel("gemma3", map[string]int{"image_size": 896, "patch_size": 14})
	if value := projectorImageTokens(gemma, 2048, 2048); value != 256 {
		t.Fatalf("Gemma budget = %d", value)
	}
	pixtral := projectorModel("pixtral", map[string]int{"patch_size": 16, "preproc_image_size": 1024})
	if value := projectorImageTokens(pixtral, 1024, 768); value != 3122 {
		t.Fatalf("Pixtral budget = %d", value)
	}
	unknown := projectorModel("future", nil)
	if value := projectorImageTokens(unknown, 0, 0); value < 4096 {
		t.Fatalf("unknown budget = %d", value)
	}
}

type mediaCounterRunner struct {
	llm.LlamaServer
	count int
	calls int
}

func (r *mediaCounterRunner) ChatMediaTokenCount(context.Context, llm.ChatRequest) (int, error) {
	r.calls++
	return r.count, nil
}

func TestNativeMediaBudgetPrefersBackendCounter(t *testing.T) {
	runner := &mediaCounterRunner{count: 2100}
	count, err := nativeMediaTokens(t.Context(), projectorModel("qwen2vl", nil), runner, llm.ChatRequest{})
	if err != nil || count != 2100 || runner.calls != 1 {
		t.Fatalf("count=%d calls=%d err=%v", count, runner.calls, err)
	}
}

func TestNativeMediaBudgetReadsImageHeadersAndHonorsCancellation(t *testing.T) {
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, image.NewRGBA(image.Rect(0, 0, 224, 224))); err != nil {
		t.Fatal(err)
	}
	req := llm.ChatRequest{Messages: []api.Message{{Role: "user", Images: []api.ImageData{buffer.Bytes(), buffer.Bytes()}}}}
	m := projectorModel("qwen2vl", map[string]int{"patch_size": 14})
	count, err := nativeMediaTokens(t.Context(), m, nil, req)
	if err != nil || count != 132 {
		t.Fatalf("count=%d err=%v", count, err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := nativeMediaTokens(ctx, m, nil, req); err != context.Canceled {
		t.Fatal(err)
	}
}
