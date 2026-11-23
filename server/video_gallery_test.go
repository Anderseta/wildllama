package server

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestVideoGallerySurvivesHistoryPruningAndDeletesOnlyManagedClips(t *testing.T) {
	s, v, _, _, _ := newImageTestService(t)
	router := gin.New()
	s.registerVideoRoutes(router)
	start := time.Date(2026, 11, 23, 1, 20, 0, 0, time.UTC)
	records := make([]videoRecord, 22)
	for i := range records {
		id := fmt.Sprintf("v%02d", i)
		r := videoRecord{ID: id, Created: start.Add(time.Duration(i) * time.Minute), Frames: 5, FPS: 16, Path: filepath.Join(v.root, "output", id, "video.webm")}
		require.NoError(t, os.MkdirAll(filepath.Dir(r.Path), 0700))
		require.NoError(t, os.WriteFile(r.Path, []byte("owned clip"), 0600))
		require.NoError(t, writeImageJSON(v.videoRecordPath(id), storedVideo{videoRecord: r, Details: &videoDetails{Prompt: "test"}}))
		records[i] = r
	}
	day := "/api/videos/gallery?tz=America%2FSao_Paulo"
	get := func(method, path string) []byte {
		resp := imageTestHTTP(router, method, path, nil, "")
		require.Equal(t, 200, resp.Code, resp.Body.String())
		return resp.Body.Bytes()
	}
	var dates struct{ Days []imageGalleryDay }
	require.NoError(t, json.Unmarshal(get("GET", day), &dates))
	require.Len(t, dates.Days, 1)
	d := dates.Days[0]
	require.Equal(t, "2026-11-22", d.Day)
	day += "&day=" + d.Day
	var page struct {
		Videos []videoRecord
		Next   string
	}
	require.NoError(t, json.Unmarshal(get("GET", day), &page))
	require.Len(t, page.Videos, 20)
	require.Equal(t, "v21", page.Videos[0].ID)
	require.NoError(t, json.Unmarshal(get("GET", day+"&before="+url.QueryEscape(page.Next)), &page))
	require.Len(t, page.Videos, 2)
	side := filepath.Join(filepath.Dir(records[0].Path), "kept.txt")
	require.NoError(t, os.WriteFile(side, []byte("user"), 0600))
	get("DELETE", "/api/videos/records/v00")
	require.FileExists(t, side)
	require.NoFileExists(t, records[0].Path)
	require.Equal(t, 409, imageTestHTTP(router, "DELETE", day+"&token="+d.DeleteToken, nil, "").Code)
	require.NoError(t, json.Unmarshal(get("GET", "/api/videos/gallery?tz=America%2FSao_Paulo"), &dates))
	get("DELETE", day+"&token="+dates.Days[0].DeleteToken)
	require.NoFileExists(t, records[21].Path)
	require.Empty(t, v.videoArchive)
}
