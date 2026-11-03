package server

import (
	"encoding/json"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/imagegen"
	"github.com/stretchr/testify/require"
)

func TestImageLoRAFilesRequirePinnedHashAndSafeSafetensorsPath(t *testing.T) {
	valid := huggingFaceTestFile("adapters/chosen.safetensors")
	source := huggingFaceRepository{SHA: strings.Repeat("b", 40), Files: []huggingFaceFile{valid, huggingFaceTestFile("weights.gguf"), huggingFaceTestFile("../escape.safetensors"), {Filename: "unverified.safetensors", Size: 100}}}
	result, err := imageLoRAFilesFromRepository("owner/lora", source)
	require.NoError(t, err)
	require.Len(t, result.Files, 1)
	require.Equal(t, valid.Filename, result.Files[0].Filename)
	require.Equal(t, source.SHA, result.Files[0].Revision)
	require.Equal(t, valid.LFS.SHA256, result.Files[0].SHA256)
	source.SHA = "main"
	_, err = imageLoRAFilesFromRepository("owner/lora", source)
	require.Error(t, err)
	for _, candidate := range []struct{ repo, revision, file string }{{"https://other.invalid/model", strings.Repeat("b", 40), "adapter.safetensors"}, {"owner/lora", "main", "adapter.safetensors"}, {"owner/lora", strings.Repeat("b", 40), "../escape.safetensors"}, {"owner/lora", strings.Repeat("b", 40), "code.py"}} {
		require.Error(t, validateImageLoRASource(candidate.repo, candidate.revision, candidate.file, true))
	}
}

func TestImageLoRAInstallEnqueuesPinnedSelectionWithoutGPUOrDownload(t *testing.T) {
	_, service, router, events, _ := newImageTestService(t)
	service.transferStarted = true // inspect the I/O queue without starting a transfer
	revision := strings.Repeat("b", 40)
	file := huggingFaceTestFile("adapter.safetensors")
	t.Setenv("HF_TOKEN", "private-test-token")
	requests := 0
	mockHuggingFaceHTTP(t, func(w http.ResponseWriter, r *http.Request) {
		requests++
		require.Equal(t, "huggingface.co", r.URL.Host)
		require.Equal(t, "Bearer private-test-token", r.Header.Get("Authorization"))
		require.True(t, r.URL.Path == "/api/models/owner/lora" || r.URL.Path == "/api/models/owner/lora/revision/"+revision)
		require.NoError(t, json.NewEncoder(w).Encode(huggingFaceRepository{SHA: revision, Files: []huggingFaceFile{file}}))
	})
	response := imageTestHTTP(router, "GET", "/api/images/loras/files?repository="+url.QueryEscape("owner/lora"), nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	require.NotContains(t, response.Body.String(), "private-test-token")
	response = imageTestHTTP(router, "GET", "/api/images/loras/files?repository=owner%2Flora", nil, "")
	require.Equal(t, 200, response.Code)
	require.Equal(t, 1, requests, "metadata polling uses bounded cache")
	request := imageLoRAInstallRequest{Repository: "owner/lora", Revision: revision, Filename: file.Filename, SHA256: file.LFS.SHA256, RequestID: "download-lora"}
	body, err := json.Marshal(request)
	require.NoError(t, err)
	response = imageTestHTTP(router, "POST", "/api/images/loras/install", body, "application/json")
	require.Equal(t, 202, response.Code, response.Body.String())
	require.Len(t, service.jobs, 1)
	require.Equal(t, "install_lora", service.jobs[0].Kind)
	require.Equal(t, file.LFS.SHA256, service.jobs[0].LoRA.SHA256)
	require.NotContains(t, response.Body.String(), "private-test-token")
	require.Empty(t, events.snapshot())
	count := requests
	response = imageTestHTTP(router, "POST", "/api/images/loras/install", body, "application/json")
	require.Equal(t, 202, response.Code)
	require.Len(t, service.jobs, 1)
	require.Equal(t, count, requests)
	request.RequestID = "wrong-checksum"
	request.SHA256 = strings.Repeat("0", 64)
	body, err = json.Marshal(request)
	require.NoError(t, err)
	response = imageTestHTTP(router, "POST", "/api/images/loras/install", body, "application/json")
	require.Equal(t, 400, response.Code)
	require.Len(t, service.jobs, 1)
}

func TestImageLoRALibraryIncludesSavedLocalAdaptersAndDeduplicatesPaths(t *testing.T) {
	_, service, router, events, _ := newImageTestService(t)
	path := filepath.Join(t.TempDir(), "adapter.safetensors")
	imageLocalTensorFixture(t, path, map[string]any{
		"transformer_blocks.0.attn.to_q.lora_A.default.weight": map[string]any{"dtype": "BF16", "shape": []int{1, 1}, "data_offsets": []int{0, 2}},
		"transformer_blocks.0.attn.to_q.lora_B.default.weight": map[string]any{"dtype": "BF16", "shape": []int{1, 1}, "data_offsets": []int{2, 4}},
	}, make([]byte, 4))
	service.config.Profiles[0].LoRAs = []imagegen.LoRA{{Path: path, Scale: 1}, {Path: path, Scale: 0}}
	response := imageTestHTTP(router, "GET", "/api/images/loras", nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	var result struct {
		LoRAs []imagegen.LoRAAsset `json:"loras"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &result))
	require.Len(t, result.LoRAs, 1)
	require.True(t, result.LoRAs[0].Ready)
	require.Equal(t, path, result.LoRAs[0].Path)
	require.Equal(t, "local", result.LoRAs[0].Origin)
	require.Greater(t, result.LoRAs[0].Size, int64(0))
	require.Empty(t, events.snapshot())
}
