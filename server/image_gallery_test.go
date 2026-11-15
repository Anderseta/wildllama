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
	require.Equal(t, []imageGalleryDay{{Day: "2026-11-15", Count: 1}, {Day: "2026-11-14", Count: 42}}, dates.Days)
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
