package server

import (
	"encoding/json"
	"github.com/stretchr/testify/require"
	"os"
	"path/filepath"
	"testing"
)

func deletionFixture(t *testing.T, v *imageService, id, filename string, input bool) imageAssetRecord {
	t.Helper()
	a := imageAssetRecord{imageAsset: imageAsset{AssetID: id + "/" + filename, JobID: id, Filename: filename, SessionID: "chat-a"}, OutputRoot: v.config.OutputDir, Input: input}
	a.Path = filepath.Join(a.OutputRoot, a.relativePath())
	require.NoError(t, os.MkdirAll(filepath.Dir(a.Path), 0700))
	require.NoError(t, os.WriteFile(a.Path, []byte("owned image fixture"), 0600))
	v.assets[a.AssetID] = a
	return a
}

func TestImageAssetDeleteLastResultCleansFolderAndSurvivesRestart(t *testing.T) {
	_, v, r, _, _ := newImageTestService(t)
	a := deletionFixture(t, v, "output", "image.png", false)
	input := deletionFixture(t, v, "input", "image.png", true)
	a.ReferenceAssetIDs = []string{input.AssetID}
	v.assets[a.AssetID] = a
	v.jobs = []*imageJob{{ID: a.JobID, Status: "completed", Result: &a.imageAsset, Results: []imageAsset{a.imageAsset}}}
	require.NoError(t, os.WriteFile(filepath.Join(filepath.Dir(a.Path), "preview.tmp"), []byte("owned thumbnail"), 0600))
	response := imageTestHTTP(r, "DELETE", "/api/images/assets/"+a.AssetID, nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	require.NoDirExists(t, filepath.Dir(a.Path))
	require.NoDirExists(t, filepath.Dir(input.Path))
	require.Empty(t, v.assets)
	require.Nil(t, v.jobs[0].Result)
	require.Contains(t, v.jobs[0].DeletedAssetIDs, a.AssetID)
	require.Equal(t, 200, imageTestHTTP(r, "DELETE", "/api/images/assets/"+a.AssetID, nil, "").Code)
	t.Setenv("WILDLLAMA_IMAGES_DIR", v.root)
	restarted := &Server{}
	t.Cleanup(restarted.stopImageJobs)
	loaded := restarted.imageService()
	require.Empty(t, loaded.assets)
	require.Nil(t, loaded.jobs[0].Result)
	require.Empty(t, loaded.jobs[0].Results)
	require.NoError(t, loaded.assetDeleteErr)
}

func TestImageAssetDeletePreservesBatchSiblingAndSharedInput(t *testing.T) {
	_, v, r, _, _ := newImageTestService(t)
	a := deletionFixture(t, v, "batch", "image.png", false)
	b := deletionFixture(t, v, "batch", "image-2.png", false)
	ref := deletionFixture(t, v, "shared", "image.png", true)
	a.ReferenceAssetIDs = []string{ref.AssetID}
	v.assets[a.AssetID] = a
	v.assets[b.AssetID] = b
	v.jobs = []*imageJob{{ID: a.JobID, Status: "completed", Result: &a.imageAsset, Results: []imageAsset{a.imageAsset, b.imageAsset}, Generation: &imageGenerationSummary{ReferenceAssetIDs: []string{ref.AssetID}}}}
	require.Equal(t, 200, imageTestHTTP(r, "DELETE", "/api/images/assets/"+a.AssetID, nil, "").Code)
	require.NoFileExists(t, a.Path)
	require.FileExists(t, b.Path)
	require.FileExists(t, ref.Path)
	require.Equal(t, b.AssetID, v.jobs[0].Result.AssetID)
	require.Len(t, v.jobs[0].Results, 1)
	require.Equal(t, 200, imageTestHTTP(r, "DELETE", "/api/images/assets/"+b.AssetID, nil, "").Code)
	require.NoDirExists(t, filepath.Dir(b.Path))
	require.NoDirExists(t, filepath.Dir(ref.Path))
}

func TestImageAssetDeleteRejectsActiveUseOtherChatAndForgedPath(t *testing.T) {
	_, v, r, _, _ := newImageTestService(t)
	a := deletionFixture(t, v, "saved", "image.png", false)
	job := &imageJob{ID: "other", Status: "running", Generation: &imageGenerationSummary{ReferenceAssetIDs: []string{a.AssetID}}}
	v.jobs = []*imageJob{job}
	response := imageTestHTTP(r, "DELETE", "/api/images/assets/"+a.AssetID, nil, "")
	require.Equal(t, 409, response.Code, response.Body.String())
	require.FileExists(t, a.Path)
	job.Status = "completed"
	body, _ := json.Marshal(map[string]string{"session_id": "chat-b"})
	require.Equal(t, 403, imageTestHTTP(r, "DELETE", "/api/images/assets/"+a.AssetID, body, "application/json").Code)
	corrupt := a
	corrupt.Path = filepath.Join(t.TempDir(), "external.png")
	v.assets[a.AssetID] = corrupt
	require.Equal(t, 400, imageTestHTTP(r, "DELETE", "/api/images/assets/"+a.AssetID, nil, "").Code)
	require.FileExists(t, a.Path)
}

func TestImageAssetDeleteJournalFailureDoesNotUnlink(t *testing.T) {
	_, v, r, _, _ := newImageTestService(t)
	a := deletionFixture(t, v, "saved", "image.png", false)
	require.NoError(t, os.MkdirAll(filepath.Join(v.root, "deleted-assets.json"), 0700))
	response := imageTestHTTP(r, "DELETE", "/api/images/assets/"+a.AssetID, nil, "")
	require.Equal(t, 500, response.Code, response.Body.String())
	require.FileExists(t, a.Path)
}

func TestImageAssetDeleteFencesUploadReplay(t *testing.T) {
	_, v, r, _, _ := newImageTestService(t)
	a := deletionFixture(t, v, "input", "image.png", true)
	a.RequestID = "upload-a"
	v.assets[a.AssetID] = a
	response := imageTestHTTP(r, "DELETE", "/api/images/assets/"+a.AssetID, nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	response = imageTestHTTP(r, "POST", imageUploadURI(a.RequestID, "chat-a"), imageUploadFixture(t, 1), "image/png")
	require.Equal(t, 410, response.Code, response.Body.String())
	require.Empty(t, v.assets)
}
