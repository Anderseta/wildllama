package server

import (
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"net/url"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestImageGalleryPagesDurableOutputsByLocalDay(t *testing.T) {
	_, service, router, _, _ := newImageBatchTestService(t)
	const route = "/api/images/gallery?tz=America%2FSao_Paulo"
	stamp := time.Date(2026, 11, 15, 2, 30, 0, 0, time.UTC)
	for index := 0; index < 42; index++ {
		id := fmt.Sprintf("gallery-%02d", index)
		// Equal timestamps exercise the identity tie-break and cursor.
		a := imageAsset{AssetID: id + "/image.png", JobID: id, Filename: "image.png", GeneratedAtMS: stamp.UnixMilli()}
		service.assets[a.AssetID] = imageAssetRecord{imageAsset: a, Generation: &imageGenerationMetadata{Prompt: "private"}}
	}
	legacy := deletionFixture(t, service, "legacy", "image.png", false)
	legacy.MIMEType = "image/png"
	f, err := os.Create(legacy.Path)
	require.NoError(t, err)
	require.NoError(t, png.Encode(f, image.NewRGBA(image.Rect(0, 0, 1, 1))))
	require.NoError(t, f.Close())
	require.NoError(t, os.Chtimes(legacy.Path, stamp, stamp.Add(2*time.Hour)))
	service.assets[legacy.AssetID] = legacy
	input := deletionFixture(t, service, "input-gallery", "image.png", true)
	input.GeneratedAtMS = stamp.UnixMilli()
	service.assets[input.AssetID] = input
	service.deletedSessions = map[string]bool{"removed": true}
	removed := imageAsset{AssetID: "removed/image.png", JobID: "removed", Filename: "image.png", SessionID: "removed", GeneratedAtMS: stamp.UnixMilli()}
	service.assets[removed.AssetID] = imageAssetRecord{imageAsset: removed}
	require.Empty(t, service.jobs, "all images must be discovered independently of job history")

	response := imageTestHTTP(router, "GET", route, nil, "")
	require.Equal(t, 200, response.Code, response.Body.String())
	var dates struct{ Days []imageGalleryDay }
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &dates))
	require.Len(t, dates.Days, 2)
	require.Equal(t, "2026-11-15", dates.Days[0].Day)
	require.Equal(t, 1, dates.Days[0].Count)
	require.Equal(t, "2026-11-14", dates.Days[1].Day)
	require.Equal(t, 42, dates.Days[1].Count)
	require.Len(t, dates.Days[1].DeleteToken, 64)
	loaded := &imageService{root: service.root}
	require.NoError(t, loaded.loadImageCatalog())
	require.Equal(t, stamp.Add(2*time.Hour).UnixMilli(), loaded.assets[legacy.AssetID].GeneratedAtMS)

	var ids []string
	cursor := ""
	for _, count := range []int{20, 20, 2} {
		response := imageTestHTTP(router, "GET", route+"&day=2026-11-14&before="+url.QueryEscape(cursor), nil, "")
		require.Equal(t, 200, response.Code, response.Body.String())
		require.NotContains(t, response.Body.String(), "private")
		var page struct {
			Images []imageGalleryItem
			Next   string
		}
		require.NoError(t, json.Unmarshal(response.Body.Bytes(), &page))
		require.Len(t, page.Images, count)
		for _, a := range page.Images {
			ids = append(ids, a.AssetID)
		}
		cursor = page.Next
		if len(ids) == 20 {
			last := page.Images[len(page.Images)-1].AssetID
			delete(service.assets, last)
			require.NoError(t, service.persistImageAssetsLocked(nil, []string{last}))
		}
	}
	require.Empty(t, cursor)
	for index, id := range ids {
		require.Equal(t, fmt.Sprintf("gallery-%02d/image.png", 41-index), id)
	}
	require.Equal(t, 400, imageTestHTTP(router, "GET", route+"&day=2026-11-14&before=invalid", nil, "").Code)
}

func TestImageGalleryDeleteDayChecksTheWholeSelectionAndCanRetry(t *testing.T) {
	_, service, router, _, _ := newImageBatchTestService(t)
	const zone = "America/Sao_Paulo"
	const day = "2026-11-14"
	stamp := time.Date(2026, 11, 15, 2, 30, 0, 0, time.UTC)
	ids := []string{}
	var outputs []imageAssetRecord
	for i := 0; i < 23; i++ {
		asset := deletionFixture(t, service, fmt.Sprintf("selected-%d", i), "image.png", false)
		asset.GeneratedAtMS = stamp.UnixMilli()
		service.assets[asset.AssetID] = asset
		ids = append(ids, asset.AssetID)
		outputs = append(outputs, asset)
	}
	other := deletionFixture(t, service, "other-day", "image.png", false)
	other.GeneratedAtMS = stamp.Add(time.Hour).UnixMilli()
	service.assets[other.AssetID] = other
	ref := deletionFixture(t, service, "reference-for-other", "image.png", true)
	outputs[0].ReferenceAssetIDs = []string{ref.AssetID}
	service.assets[outputs[0].AssetID] = outputs[0]
	other.ReferenceAssetIDs = []string{ref.AssetID}
	service.assets[other.AssetID] = other

	request := map[string]any{"day": day, "tz": zone, "count": len(ids), "delete_token": imageGalleryDeleteToken(day, zone, ids)}
	remove := func() int {
		body, err := json.Marshal(request)
		require.NoError(t, err)
		return imageTestHTTP(router, "DELETE", "/api/images/gallery/day", body, "application/json").Code
	}
	active := &imageJob{ID: "busy", Status: "running", Generation: &imageGenerationSummary{ReferenceAssetIDs: []string{outputs[0].AssetID}}}
	service.jobs = []*imageJob{active}
	require.Equal(t, 409, remove())
	require.FileExists(t, outputs[len(outputs)-1].Path)
	service.jobs = nil
	request["count"] = 20
	require.Equal(t, 409, remove(), "a partially loaded page is not the whole day")
	request["count"] = len(ids)
	require.Equal(t, 200, remove())
	for _, asset := range outputs {
		require.NoFileExists(t, asset.Path)
	}
	require.FileExists(t, other.Path)
	require.FileExists(t, ref.Path)
	newAsset := deletionFixture(t, service, "new-same-day", "image.png", false)
	newAsset.GeneratedAtMS = stamp.UnixMilli()
	service.assets[newAsset.AssetID] = newAsset
	require.Equal(t, 200, remove(), "the same request can resume after a lost acknowledgement")
	require.FileExists(t, newAsset.Path)
}

func TestImageJobHistoryClearPreservesActiveJobsAndPublishedImages(t *testing.T) {
	_, service, router, _, _ := newImageBatchTestService(t)
	asset := deletionFixture(t, service, "saved-history", "image.png", false)
	saved := &imageJob{ID: asset.JobID, Status: "completed", Result: &asset.imageAsset}
	failed := &imageJob{ID: "old-failure", Status: "failed", Error: "old CUDA error"}
	busy := &imageJob{ID: "active", Status: "running", Progress: imageProgress{Message: "live status"}}
	service.jobs = []*imageJob{saved, failed, busy}
	require.NoError(t, service.persistJobsLocked())
	require.Equal(t, 200, imageTestHTTP(router, "DELETE", "/api/images/jobs/history", nil, "").Code)
	response := imageTestHTTP(router, "GET", "/api/images/jobs?summary=true", nil, "")
	require.Equal(t, 200, response.Code)
	require.Contains(t, response.Body.String(), "live status")
	require.NotContains(t, response.Body.String(), "old-failure")
	require.Equal(t, 200, imageTestHTTP(router, "GET", "/api/images/jobs/"+asset.JobID, nil, "").Code)
	require.FileExists(t, asset.Path)
	require.True(t, service.jobs[1].HistoryCleared)
	require.Empty(t, service.jobs[1].Error)
}
