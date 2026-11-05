package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestImageReferencesAPIToRunnerPreservesOneAndFour(t *testing.T) {
	for _, count := range []int{1, 4} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			_, service, router, _, runner := newImageBatchTestService(t)
			vision := filepath.Join(service.root, "vision.gguf")
			require.NoError(t, os.WriteFile(vision, append([]byte("GGUF"), make([]byte, 20)...), 0600))
			service.config.Profiles[0].VisionEncoder = vision
			var original bytes.Buffer
			require.NoError(t, png.Encode(&original, image.NewRGBA(image.Rect(0, 0, 64, 64))))
			ids, paths := make([]string, count), make([]string, count)
			for i := range ids {
				name := fmt.Sprintf("reference-%d", count-i)
				ids[i] = name + "/image.png"
				asset := imageAssetRecord{imageAsset: imageAsset{AssetID: ids[i], JobID: name, Filename: "image.png", MIMEType: "image/png", Width: 64, Height: 64}, OutputRoot: service.config.OutputDir, Input: true}
				paths[i] = filepath.Join(asset.OutputRoot, asset.relativePath())
				require.NoError(t, os.MkdirAll(filepath.Dir(paths[i]), 0700))
				require.NoError(t, os.WriteFile(paths[i], original.Bytes(), 0600))
				service.assets[ids[i]] = asset
			}
			request := imageJobRequest{RequestID: "reference-order", Prompt: "Compose the references in the supplied order", ReferenceImages: ids}
			body, err := json.Marshal(request)
			require.NoError(t, err)
			response := imageTestHTTP(router, "POST", "/api/images/jobs", body, "application/json")
			require.Equal(t, 202, response.Code, response.Body.String())
			service.mu.Lock()
			jobs := append([]*imageJob(nil), service.jobs...)
			service.mu.Unlock()
			require.Len(t, jobs, 1)
			job := jobs[0]
			waitImageBatchTerminal(t, service, job)
			service.mu.Lock()
			status, jobError := job.Status, job.Error
			generation, parent := job.Generation, job.ParentAssetID
			service.mu.Unlock()
			require.Equal(t, "completed", status, jobError)
			require.Equal(t, ids, generation.ReferenceAssetIDs)
			require.Equal(t, ids[0], parent)
			runner.mu.Lock()
			callCount := len(runner.calls)
			var calls []string
			if callCount > 0 {
				calls = append([]string(nil), runner.calls[0].ReferenceImages...)
			}
			runner.mu.Unlock()
			require.Equal(t, 1, callCount)
			require.Equal(t, paths, calls)
			for _, path := range paths {
				data, err := os.ReadFile(path)
				require.NoError(t, err)
				require.Equal(t, original.Bytes(), data)
			}
		})
	}
}

func TestImageReferencesRejectFifthIncludingLegacyInput(t *testing.T) {
	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprint(legacy), func(t *testing.T) {
			_, service, router, events, runner := newImageBatchTestService(t)
			request := imageJobRequest{Prompt: "fixture", ReferenceImages: []string{"a", "b", "c", "d"}}
			if legacy {
				request.InputAssetID = "e"
			} else {
				request.ReferenceImages = append(request.ReferenceImages, "e")
			}
			body, err := json.Marshal(request)
			require.NoError(t, err)
			response := imageTestHTTP(router, "POST", "/api/images/jobs", body, "application/json")
			require.Equal(t, 400, response.Code, response.Body.String())
			require.Empty(t, service.jobs)
			require.Empty(t, runner.calls)
			require.NotContains(t, events.snapshot(), "suspend_text")
		})
	}
}
