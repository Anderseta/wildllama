package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestImageSessionDeletionRemovesInputsOutputsAndPersistsFence(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	upload := imageTestHTTP(router, "POST", imageUploadURI("own-upload", "chat-a"), imageUploadFixture(t, 1), "image/png")
	require.Equal(t, 201, upload.Code, upload.Body.String())
	input, _ := imageUploadedResponse(t, upload)
	other := imageTestHTTP(router, "POST", imageUploadURI("other-upload", "chat-b"), imageUploadFixture(t, 2), "image/png")
	require.Equal(t, 201, other.Code, other.Body.String())
	otherInput, _ := imageUploadedResponse(t, other)
	job := imageLifecycleJob(service, "")
	job.RequestID, job.SessionID = "own-generation", "chat-a"
	job, err := service.enqueue(job)
	require.NoError(t, err)
	waitImageBatchTerminal(t, service, job)
	partial := filepath.Join(service.config.OutputDir, job.ID, "partial.tmp")
	require.NoError(t, os.WriteFile(partial, []byte("owned partial"), 0600))
	deleted := imageTestHTTP(router, "DELETE", "/api/images/sessions/chat-a", nil, "")
	require.Equal(t, 200, deleted.Code, deleted.Body.String())
	require.NoDirExists(t, filepath.Dir(input.Path))
	require.NoDirExists(t, filepath.Dir(partial))
	require.FileExists(t, otherInput.Path)
	require.Empty(t, service.jobs)
	require.Len(t, service.assets, 1)
	for _, session := range []string{"chat-a", "chat-b"} {
		replayed := imageTestHTTP(router, "POST", imageUploadURI("own-upload", session), imageUploadFixture(t, 1), "image/png")
		require.Equal(t, 410, replayed.Code, replayed.Body.String())
	}
	_, err = service.enqueue(&imageJob{SessionID: "chat-b", RequestID: "own-generation"})
	require.ErrorContains(t, err, "deleted")
	t.Setenv("WILDLLAMA_IMAGES_DIR", service.root)
	restarted := &Server{}
	t.Cleanup(restarted.stopImageJobs)
	loaded := restarted.imageService()
	loaded.mu.Lock()
	err = loaded.checkImageSessionLocked("chat-a", "new-after-restart")
	loaded.mu.Unlock()
	require.ErrorContains(t, err, "deleted")
	response := imageTestHTTP(mustImageRouter(t, restarted), "DELETE", "/api/images/sessions/chat-a", nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
}

func TestImageSessionDeletionCancelsActiveAndQueuedBeforeRemovingFiles(t *testing.T) {
	_, service, router, _, runner := newImageTestService(t)
	runner.entered, runner.wait = make(chan struct{}), true
	first := imageLifecycleJob(service, "")
	first.SessionID = "active-chat"
	first, err := service.enqueue(first)
	require.NoError(t, err)
	select {
	case <-runner.entered:
	case <-time.After(time.Second):
		t.Fatal("image worker did not start")
	}
	second := imageLifecycleJob(service, "")
	second.SessionID = "active-chat"
	_, err = service.enqueue(second)
	require.NoError(t, err)
	response := imageTestHTTP(router, "DELETE", "/api/images/sessions/active-chat", nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	service.mu.Lock()
	executing, jobs, assets, queued := first.executing, len(service.jobs), len(service.assets), len(service.queue)
	service.mu.Unlock()
	require.False(t, executing)
	require.Zero(t, jobs)
	require.Zero(t, assets)
	require.Zero(t, queued)
	require.NoDirExists(t, filepath.Join(service.config.OutputDir, first.ID))
}

func TestImageSessionDeletionPreservesSharedLineageAndClaimsOnlyLegacyUnknown(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	add := func(id, session string) imageAssetRecord {
		asset := imageAssetRecord{imageAsset: imageAsset{AssetID: id + "/image.png", JobID: id, SessionID: session, Filename: "image.png"}, OutputRoot: service.config.OutputDir}
		asset.Path = filepath.Join(asset.OutputRoot, asset.relativePath())
		require.NoError(t, os.MkdirAll(filepath.Dir(asset.Path), 0700))
		require.NoError(t, os.WriteFile(asset.Path, []byte("owned image fixture"), 0600))
		service.assets[asset.AssetID] = asset
		return asset
	}
	parent := add("parent", "chat-a")
	secondReference := add("second-reference", "chat-a")
	child := add("child", "chat-b")
	child.ParentAssetID = parent.AssetID
	child.ReferenceAssetIDs = []string{parent.AssetID, secondReference.AssetID}
	service.assets[child.AssetID] = child
	legacy := add("legacy", "")
	unknown := add("unknown", "")
	body, err := json.Marshal(map[string]any{"legacy_asset_ids": []string{legacy.AssetID, child.AssetID}})
	require.NoError(t, err)
	response := imageTestHTTP(router, "DELETE", "/api/images/sessions/chat-a", body, "application/json")
	require.Equal(t, 200, response.Code, response.Body.String())
	require.NoFileExists(t, legacy.Path)
	for _, asset := range []imageAssetRecord{parent, secondReference, child, unknown} {
		require.FileExists(t, asset.Path)
	}
	response = imageTestHTTP(router, "DELETE", "/api/images/sessions/chat-b", nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	for _, asset := range []imageAssetRecord{parent, secondReference, child} {
		require.NoDirExists(t, filepath.Dir(asset.Path))
	}
	require.FileExists(t, unknown.Path)
}

type imageDeletionUploadReader struct {
	bytes   *bytes.Reader
	read    chan struct{}
	release chan struct{}
	first   bool
}

func (r *imageDeletionUploadReader) Read(p []byte) (int, error) {
	if !r.first {
		r.first = true
		close(r.read)
		<-r.release
	}
	return r.bytes.Read(p)
}

func TestImageSessionDeletionJoinsInFlightUploadAndRejectsLatePublication(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	reader := &imageDeletionUploadReader{bytes: bytes.NewReader(imageUploadFixture(t, 1)), read: make(chan struct{}), release: make(chan struct{})}
	request := httptest.NewRequest("POST", imageUploadURI("slow-upload", "slow-chat"), io.NopCloser(reader))
	request.Header.Set("Content-Type", "image/png")
	uploaded := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		response := httptest.NewRecorder()
		router.ServeHTTP(response, request)
		uploaded <- response
	}()
	<-reader.read
	deleted := make(chan *httptest.ResponseRecorder, 1)
	go func() { deleted <- imageTestHTTP(router, "DELETE", "/api/images/sessions/slow-chat", nil, "") }()
	require.Eventually(t, func() bool { service.mu.Lock(); defer service.mu.Unlock(); return service.deletedSessions["slow-chat"] }, time.Second, time.Millisecond)
	select {
	case <-deleted:
		t.Fatal("deletion returned before the upload released its staging files")
	default:
	}
	close(reader.release)
	response := <-uploaded
	require.Equal(t, 410, response.Code, response.Body.String())
	response = <-deleted
	require.Equal(t, 200, response.Code, response.Body.String())
	require.Empty(t, service.assets)
	entries, err := os.ReadDir(filepath.Join(service.config.OutputDir, "inputs"))
	require.NoError(t, err)
	require.Empty(t, entries)
	response = imageTestHTTP(router, "POST", imageUploadURI("slow-upload", "another-chat"), imageUploadFixture(t, 1), "image/png")
	require.Equal(t, 410, response.Code, response.Body.String())
}

func TestImageSessionDeletionRejectsBlankAndUnsafePaths(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	response := imageTestHTTP(router, "DELETE", "/api/images/sessions/%20", nil, "")
	require.Equal(t, 400, response.Code)
	require.Empty(t, service.deletedSessions)
	require.NoError(t, service.checkImageSessionLocked(""))
	outside := filepath.Join(t.TempDir(), "preserve.png")
	require.NoError(t, os.WriteFile(outside, []byte("original"), 0600))
	service.assets["bad/image.png"] = imageAssetRecord{imageAsset: imageAsset{AssetID: "bad/image.png", JobID: "bad", Filename: "image.png", SessionID: "unsafe-chat", Path: outside}, OutputRoot: service.config.OutputDir}
	response = imageTestHTTP(router, "DELETE", "/api/images/sessions/unsafe-chat", nil, "")
	require.Equal(t, 500, response.Code, response.Body.String())
	require.FileExists(t, outside)
	require.Error(t, removeImageOwnedPath(service.config.OutputDir, "../preserve.png", true))
}

func TestImageSessionDeletionDoesNotFollowDirectoryLinks(t *testing.T) {
	root, outside := t.TempDir(), t.TempDir()
	original := filepath.Join(outside, "image.png")
	require.NoError(t, os.WriteFile(original, []byte("original"), 0600))
	if err := os.Symlink(outside, filepath.Join(root, "linked")); err != nil {
		t.Skipf("directory links unavailable: %v", err)
	}
	require.Error(t, removeImageOwnedPath(root, filepath.Join("linked", "image.png"), false))
	require.FileExists(t, original)
	require.NoError(t, removeImageOwnedPath(root, "linked", true))
	require.FileExists(t, original)
	_, err := os.Lstat(filepath.Join(root, "linked"))
	require.True(t, errors.Is(err, os.ErrNotExist))
}

func TestImageSessionDeletionRetainsExecutingTerminalJobDuringPruning(t *testing.T) {
	_, service, _, _, _ := newImageTestService(t)
	active := &imageJob{ID: "still-cleaning-up", Kind: "generate", SessionID: "chat-a", Status: "completed", executing: true}
	service.jobs = append(service.jobs, active)
	for range 200 {
		service.jobs = append(service.jobs, &imageJob{ID: newImageID(), Status: "completed"})
	}
	require.NoError(t, service.persistJobsLocked())
	require.Len(t, service.jobs, 200)
	require.Same(t, active, service.jobs[0])
}

func TestImageSessionDeletionFindsFailedPartialDirectoryAfterHistoryPruningAndRestart(t *testing.T) {
	_, service, _, _, _ := newImageTestService(t)
	job := &imageJob{ID: "pruned-partial", Kind: "generate", SessionID: "chat-a", Config: cloneImageConfig(service.config)}
	require.NoError(t, service.recordImageDirectoryLocked(job))
	directory := filepath.Join(service.config.OutputDir, job.ID)
	require.NoError(t, os.MkdirAll(directory, 0700))
	require.NoError(t, os.WriteFile(filepath.Join(directory, "partial.tmp"), []byte("partial"), 0600))
	require.NoError(t, writeImageJSON(filepath.Join(service.root, "jobs.json"), []*imageJob{}))
	t.Setenv("WILDLLAMA_IMAGES_DIR", service.root)
	restarted := &Server{}
	t.Cleanup(restarted.stopImageJobs)
	response := imageTestHTTP(mustImageRouter(t, restarted), "DELETE", "/api/images/sessions/chat-a", nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	require.NoDirExists(t, directory)
	require.Empty(t, restarted.imageService().ownedDirectories)
}

func TestImageSessionDeletionJournalCorruptionFailsClosed(t *testing.T) {
	_, service, _, _, _ := newImageTestService(t)
	require.NoError(t, os.WriteFile(filepath.Join(service.root, "deleted-sessions.json"), []byte(`null`), 0600))
	t.Setenv("WILDLLAMA_IMAGES_DIR", service.root)
	restarted := &Server{}
	t.Cleanup(restarted.stopImageJobs)
	loaded := restarted.imageService()
	require.Error(t, loaded.checkImageSessionLocked("chat-a", "stale-request"))
	response := imageTestHTTP(mustImageRouter(t, restarted), "DELETE", "/api/images/sessions/chat-a", nil, "")
	require.Equal(t, 500, response.Code)
}
