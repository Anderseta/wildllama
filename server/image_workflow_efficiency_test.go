package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type blockedImageResponse struct {
	*httptest.ResponseRecorder
	entered chan struct{}
	release chan struct{}
}

func (w *blockedImageResponse) Write(p []byte) (int, error) {
	close(w.entered)
	<-w.release
	return w.ResponseRecorder.Write(p)
}

func TestImageStatusNetworkWriteDoesNotHoldServiceMutex(t *testing.T) {
	for _, path := range []string{"/api/images/jobs", "/api/images/jobs/snapshot"} {
		t.Run(path, func(t *testing.T) {
			_, service, router, _, _ := newImageTestService(t)
			job := &imageJob{ID: "snapshot", Status: "running", Phase: "original", Generation: &imageGenerationSummary{ReferenceAssetIDs: []string{"original"}}}
			service.jobs = []*imageJob{job}
			writer := &blockedImageResponse{ResponseRecorder: httptest.NewRecorder(), entered: make(chan struct{}), release: make(chan struct{})}
			done := make(chan struct{})
			go func() { router.ServeHTTP(writer, httptest.NewRequest(http.MethodGet, path, nil)); close(done) }()
			select {
			case <-writer.entered:
			case <-time.After(time.Second):
				t.Fatal("response did not start")
			}
			updated := make(chan struct{})
			go func() {
				service.update(job, "new", "live", 2, 4)
				service.mu.Lock()
				job.Generation.ReferenceAssetIDs[0] = "new"
				service.mu.Unlock()
				close(updated)
			}()
			select {
			case <-updated:
			case <-time.After(time.Second):
				close(writer.release)
				t.Fatal("HTTP writer blocked progress mutex")
			}
			close(writer.release)
			<-done
			require.Contains(t, writer.Body.String(), `"phase":"original"`)
			require.Contains(t, writer.Body.String(), `"reference_asset_ids":["original"]`)
		})
	}
}

func TestImagePendingTransferDoesNotBlockReadyGeneration(t *testing.T) {
	_, service, _, _, runner := newImageBatchTestService(t)
	// Hold the I/O worker before it dequeues, simulating a transfer awaiting
	// network capacity, without downloading files or consuming GPU memory.
	service.transferStarted = true
	transfer, err := service.enqueue(&imageJob{Kind: "install", RecipeID: "fixture", Config: cloneImageConfig(service.config)})
	require.NoError(t, err)
	job, err := service.enqueue(imageLifecycleJob(service, ""))
	require.NoError(t, err)
	waitImageBatchTerminal(t, service, job)
	service.mu.Lock()
	require.Equal(t, "queued", transfer.Status)
	require.Equal(t, "completed", job.Status)
	service.mu.Unlock()
	runner.mu.Lock()
	calls := len(runner.calls)
	runner.mu.Unlock()
	require.Equal(t, 1, calls)
}

func TestImageAssetCatalogMigrationAndIncrementalDeletion(t *testing.T) {
	root := t.TempDir()
	legacy := imageAssetRecord{imageAsset: imageAsset{AssetID: "first/image.png", JobID: "first", Filename: "image.png", SessionID: "session-a", ParentAssetID: "input/image.png"}, ReferenceAssetIDs: []string{"input/image.png"}, OutputRoot: root}
	old := map[string]imageAssetRecord{legacy.AssetID: legacy}
	data, err := json.Marshal(old)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(root, "assets.json"), data, 0600))
	service := &imageService{root: root, assets: old}
	second := legacy
	second.AssetID = "second/image.png"
	second.JobID = "second"
	service.assets[second.AssetID] = second
	require.NoError(t, service.persistImageAssetsLocked(map[string]imageAssetRecord{second.AssetID: second}, nil))
	preserved, err := os.ReadFile(filepath.Join(root, "assets.json"))
	require.NoError(t, err)
	require.Equal(t, data, preserved)
	loaded := &imageService{root: root}
	require.NoError(t, loaded.loadImageCatalog())
	require.Equal(t, service.assets, loaded.assets)
	delete(service.assets, legacy.AssetID)
	require.NoError(t, service.persistImageAssetsLocked(nil, []string{legacy.AssetID}))
	loaded = &imageService{root: root, assets: old}
	require.NoError(t, loaded.loadImageCatalog())
	require.Len(t, loaded.assets, 1)
	require.Equal(t, second, loaded.assets[second.AssetID])
	// A constraint failure must roll back every record in the transaction.
	db, err := service.openImageCatalog()
	require.NoError(t, err)
	_, err = db.Exec(`CREATE TRIGGER reject_bad BEFORE INSERT ON assets WHEN NEW.id='bad' BEGIN SELECT RAISE(ABORT,'fixture'); END`)
	require.NoError(t, err)
	require.NoError(t, db.Close())
	changed := second
	changed.SessionID = "must-roll-back"
	require.Error(t, service.persistImageAssetsLocked(map[string]imageAssetRecord{second.AssetID: changed, "bad": {}}, nil))
	require.NoError(t, loaded.loadImageCatalog())
	require.Equal(t, second, loaded.assets[second.AssetID])
}

func TestImageTerminalHistoryFailureKeepsRecoverableResult(t *testing.T) {
	root := t.TempDir()
	require.NoError(t, os.Mkdir(filepath.Join(root, "jobs.json"), 0700)) // sharing/write failure
	now := time.Now().UTC()
	job := &imageJob{ID: "recovery", Status: "completed", Phase: "completed", FinishedAt: &now, Result: &imageAsset{AssetID: "recovery/image.png"}}
	service := &imageService{root: root, jobs: []*imageJob{job}}
	service.persistImageJobStateLocked(job)
	require.NotEmpty(t, job.PersistenceError)
	restarted := &imageService{root: root, jobs: []*imageJob{{ID: "recovery", Status: "interrupted", FinishedAt: &now}}}
	restarted.recoverImageJobStates()
	require.Equal(t, "completed", restarted.jobs[0].Status)
	require.Equal(t, job.Result, restarted.jobs[0].Result)
}

func TestRuntimeActivitySnapshotSurvivesSchedulerContention(t *testing.T) {
	runner := &runnerRef{activity: &runnerActivityEntry{reporter: activityFixture{}, model: "fixture", pid: 7}}
	runner.model.Store(&Model{ShortName: "fixture"})
	scheduler := &Scheduler{loaded: map[string]*runnerRef{"fixture": runner}}
	scheduler.loadedMu.Lock()
	scheduler.publishLoadedSnapshotLocked()
	defer scheduler.loadedMu.Unlock()
	server := &Server{sched: scheduler}
	response := httptest.NewRecorder()
	context, _ := gin.CreateTestContext(response)
	server.RuntimeActivityHandler(context)
	var result struct {
		Runners []runnerActivityResponse `json:"runners"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &result))
	require.Len(t, result.Runners, 1)
	require.Equal(t, 7, result.Runners[0].PID)
}
