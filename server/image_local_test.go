package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/imagegen"

	"github.com/stretchr/testify/require"
)

func TestImageLocalCopyPreservesSourceAndVerifiesDestination(t *testing.T) {
	directory := t.TempDir()
	source := filepath.Join(directory, "source.bin")
	target := filepath.Join(directory, "copy.bin")
	data := bytes.Repeat([]byte("exact model bytes"), 8192)
	require.NoError(t, os.WriteFile(source, data, 0600))
	before, err := os.Stat(source)
	require.NoError(t, err)
	var phases []string
	digest, size, err := copyImageFile(t.Context(), source, target, before, func(phase string, _, _ int64) { phases = append(phases, phase) })
	require.NoError(t, err)
	require.Equal(t, int64(len(data)), size)
	expected := sha256.Sum256(data)
	require.Equal(t, hex.EncodeToString(expected[:]), digest)
	actual, err := os.ReadFile(target)
	require.NoError(t, err)
	require.Equal(t, data, actual)
	actual, err = os.ReadFile(source)
	require.NoError(t, err)
	require.Equal(t, data, actual)
	after, err := os.Stat(source)
	require.NoError(t, err)
	require.True(t, os.SameFile(before, after))
	require.Equal(t, before.ModTime(), after.ModTime())
	require.Contains(t, phases, "copying")
	require.Contains(t, phases, "verifying")
}

func imageLocalTensorFixture(t *testing.T, path string, tensors map[string]any, payload []byte) {
	t.Helper()
	header, err := json.Marshal(tensors)
	require.NoError(t, err)
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	data = append(data, payload...)
	require.NoError(t, os.WriteFile(path, data, 0600))
}

func TestImageLocalAsyncCopyRegistersBaseProfileAndSharedVisionWithoutGPU(t *testing.T) {
	_, service, router, events, _ := newImageTestService(t)
	directory := t.TempDir()
	diffusion := filepath.Join(directory, "diffusion.gguf")
	vae := filepath.Join(directory, "vae.gguf")
	for _, path := range []string{diffusion, vae} {
		require.NoError(t, os.WriteFile(path, append([]byte("GGUF"), make([]byte, 28)...), 0600))
	}
	encoder := filepath.Join(directory, "encoder.safetensors")
	imageLocalTensorFixture(t, encoder, map[string]any{
		"model.visual.patch_embed.proj.weight":  map[string]any{"dtype": "F32", "shape": []int{1, 1}, "data_offsets": []int{0, 4}},
		"model.visual.merger.linear_fc2.weight": map[string]any{"dtype": "F32", "shape": []int{1, 1}, "data_offsets": []int{4, 8}},
		"model.embed_tokens.weight":             map[string]any{"dtype": "F32", "shape": []int{1, 1}, "data_offsets": []int{8, 12}},
	}, make([]byte, 12))
	lora := filepath.Join(directory, "requested-lora.safetensors")
	imageLocalTensorFixture(t, lora, map[string]any{
		"transformer_blocks.0.attn.to_q.lora_A.default.weight": map[string]any{"dtype": "BF16", "shape": []int{1, 1}, "data_offsets": []int{0, 2}},
		"transformer_blocks.0.attn.to_q.lora_B.default.weight": map[string]any{"dtype": "BF16", "shape": []int{1, 1}, "data_offsets": []int{2, 4}},
	}, make([]byte, 4))
	components := []imageCopyComponent{{Role: "diffusion_model", Path: diffusion}, {Role: "text_encoder", Path: encoder}, {Role: "vae", Path: vae}, {Role: "lora", Path: lora}}
	body, err := json.Marshal(imageCopyRequest{Name: "Local fixture", RequestID: "local-copy", Components: components})
	require.NoError(t, err)
	response := imageTestHTTP(router, "POST", "/api/images/models/copy", body, "application/json")
	require.Equal(t, 202, response.Code, response.Body.String())
	var reply struct {
		Job imageJob `json:"job"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &reply))
	require.Eventually(t, func() bool {
		service.mu.Lock()
		defer service.mu.Unlock()
		return len(service.jobs) > 0 && imageTerminal(service.jobs[0].Status)
	}, 2*time.Second, 5*time.Millisecond)
	service.mu.Lock()
	job := service.jobs[0]
	status, errorMessage, modelID := job.Status, job.Error, job.ModelID
	config := cloneImageConfig(service.config)
	service.mu.Unlock()
	require.Equal(t, "completed", status, errorMessage)
	require.Empty(t, events.snapshot(), "local copy must never invoke text/GPU orchestration")
	models, _, err := imagegen.Catalog(service.config.ModelsDir)
	require.NoError(t, err)
	require.Len(t, models, 1)
	model := models[0]
	require.True(t, model.Ready)
	require.Equal(t, modelID, model.ID)
	require.Equal(t, 25, model.Profile.Steps)
	require.Equal(t, 1.0, model.Profile.CFGScale)
	require.Empty(t, model.Profile.Sigmas)
	require.Equal(t, "simple", model.Profile.Scheduler)
	require.True(t, model.Profile.StageComponents)
	require.Equal(t, model.Profile.TextEncoder, model.Profile.VisionEncoder, "embedded vision reuses one copied file")
	require.Len(t, model.Profile.LoRAs, 1)
	require.Equal(t, 1.0, model.Profile.LoRAs[0].Scale)
	entries, err := os.ReadDir(model.Directory)
	require.NoError(t, err)
	require.Len(t, entries, 6, "four weights plus manifest and verification receipt")
	require.Len(t, config.Profiles, 2)
	require.Equal(t, 25, config.Profiles[1].Steps)
	require.Len(t, config.Profiles[1].LoRAs, 1)
	for _, component := range components {
		_, err := os.Stat(component.Path)
		require.NoError(t, err, "source originals are preserved")
	}
	response = imageTestHTTP(router, "POST", "/api/images/models/copy", body, "application/json")
	require.Equal(t, 202, response.Code)
	service.mu.Lock()
	require.Len(t, service.jobs, 1)
	service.mu.Unlock()
}

func TestImageLocalCopyCancellationRemovesOnlyItsPartialCopy(t *testing.T) {
	directory := t.TempDir()
	source := filepath.Join(directory, "source.bin")
	target := filepath.Join(directory, "copy.bin")
	data := bytes.Repeat([]byte{1, 2, 3, 4}, 1<<20)
	require.NoError(t, os.WriteFile(source, data, 0600))
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	_, _, err := copyImageFile(ctx, source, target, nil, func(string, int64, int64) { cancel() })
	require.ErrorIs(t, err, context.Canceled)
	_, err = os.Stat(target)
	require.True(t, os.IsNotExist(err))
	actual, err := os.ReadFile(source)
	require.NoError(t, err)
	require.Equal(t, data, actual)
}

func TestImageLocalCopyRejectsChangedSourceAndExistingDestination(t *testing.T) {
	directory := t.TempDir()
	source := filepath.Join(directory, "source.bin")
	target := filepath.Join(directory, "copy.bin")
	require.NoError(t, os.WriteFile(source, []byte("old"), 0600))
	before, err := os.Stat(source)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(source, []byte("changed size"), 0600))
	_, _, err = copyImageFile(t.Context(), source, target, before, nil)
	require.ErrorContains(t, err, "changed after selection")
	_, err = os.Stat(target)
	require.True(t, os.IsNotExist(err))
	require.NoError(t, os.WriteFile(target, []byte("preserve existing file"), 0600))
	_, _, err = copyImageFile(t.Context(), source, target, nil, nil)
	require.Error(t, err)
	actual, err := os.ReadFile(target)
	require.NoError(t, err)
	require.Equal(t, "preserve existing file", string(actual))
}

func TestImageLocalInventoryIsBoundedAndDoesNotWriteOrDownload(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	directory := filepath.Join(t.TempDir(), "vae")
	require.NoError(t, os.Mkdir(directory, 0700))
	path := filepath.Join(directory, "qwen_image_2.1_vae_bf16.safetensors")
	require.NoError(t, os.WriteFile(path, []byte("invalid fixture"), 0600))
	body, err := json.Marshal(map[string]string{"directory": directory})
	require.NoError(t, err)
	response := imageTestHTTP(router, "POST", "/api/images/models/inventory", body, "application/json")
	require.Equal(t, 200, response.Code, response.Body.String())
	var result struct {
		Candidates []imageLocalCandidate `json:"candidates"`
		Truncated  bool                  `json:"truncated"`
		Profile    struct {
			Steps           int     `json:"steps"`
			CFGScale        float64 `json:"cfg_scale"`
			Sigmas          string  `json:"sigmas"`
			StageComponents bool    `json:"stage_components"`
		} `json:"recommended_profile"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &result))
	require.Len(t, result.Candidates, 1)
	require.Equal(t, "vae", result.Candidates[0].Role)
	require.Equal(t, "unsupported", result.Candidates[0].Compatibility)
	require.NotEmpty(t, result.Candidates[0].Hint)
	require.Equal(t, 25, result.Profile.Steps)
	require.Equal(t, 1.0, result.Profile.CFGScale)
	require.Empty(t, result.Profile.Sigmas)
	require.True(t, result.Profile.StageComponents)
	require.False(t, result.Truncated)
	require.Empty(t, service.jobs)
	_, err = os.Stat(service.config.ModelsDir)
	require.True(t, os.IsNotExist(err))
	response = imageTestHTTP(router, "POST", "/api/images/models/inventory", []byte(`{"directory":"relative"}`), "application/json")
	require.Equal(t, 400, response.Code)
}

func TestImageLocalCopyInvalidSelectionCannotEnqueueOrCreateModels(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	response := imageTestHTTP(router, "POST", "/api/images/models/copy", []byte(`{"components":[]}`), "application/json")
	require.Equal(t, 400, response.Code)
	require.Empty(t, service.jobs)
	_, err := os.Stat(service.config.ModelsDir)
	require.True(t, os.IsNotExist(err))
}

func TestImageConfigSnapshotsOwnTheirLoRASlices(t *testing.T) {
	config := defaultImageConfig(t.TempDir())
	config.Profiles[0].LoRAs = []imagegen.LoRA{{Path: "original.safetensors", Scale: 1}}
	snapshot := cloneImageConfig(config)
	snapshot.Profiles[0].LoRAs[0].Path = "changed.safetensors"
	snapshot.Profiles[0].LoRAs[0].Scale = 0
	require.Equal(t, "original.safetensors", config.Profiles[0].LoRAs[0].Path)
	require.Equal(t, 1.0, config.Profiles[0].LoRAs[0].Scale)
}
