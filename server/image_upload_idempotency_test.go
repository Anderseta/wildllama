package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func imageUploadFixture(t *testing.T, red uint8) []byte {
	t.Helper()
	var body bytes.Buffer
	fixture := image.NewRGBA(image.Rect(0, 0, 64, 32))
	fixture.Set(0, 0, color.RGBA{R: red, A: 255})
	require.NoError(t, png.Encode(&body, fixture))
	return body.Bytes()
}

func imageUploadURI(requestID, sessionID string) string {
	return "/api/images/assets?" + url.Values{"request_id": {requestID}, "session_id": {sessionID}}.Encode()
}

func imageUploadedResponse(t *testing.T, response *httptest.ResponseRecorder) (imageAsset, bool) {
	t.Helper()
	var value struct {
		Asset    imageAsset `json:"asset"`
		Replayed bool       `json:"replayed"`
	}
	require.NoError(t, json.Unmarshal(response.Body.Bytes(), &value))
	return value.Asset, value.Replayed
}

type imageLostUploadACK struct {
	header http.Header
	status int
}

func (w *imageLostUploadACK) Header() http.Header    { return w.header }
func (w *imageLostUploadACK) WriteHeader(status int) { w.status = status }
func (w *imageLostUploadACK) Write([]byte) (int, error) {
	return 0, errors.New("simulated lost upload acknowledgement")
}

func TestImageUploadIdempotencyLostACKAndRestart(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	body := imageUploadFixture(t, 220)
	uri := imageUploadURI("stable-upload-key", "chat-a")
	lost := &imageLostUploadACK{header: make(http.Header)}
	request := httptest.NewRequest("POST", uri, bytes.NewReader(body))
	request.Header.Set("Content-Type", "image/png")
	router.ServeHTTP(lost, request)
	require.Equal(t, 201, lost.status)
	require.Len(t, service.assets, 1)
	first := imageTestHTTP(router, "POST", uri, body, "image/png")
	require.Equal(t, 200, first.Code, first.Body.String())
	asset, replayed := imageUploadedResponse(t, first)
	require.True(t, replayed)
	require.Equal(t, "stable-upload-key", asset.RequestID)
	require.Equal(t, "chat-a", asset.SessionID)
	require.Len(t, service.assets[asset.AssetID].UploadFingerprint, 64)
	t.Setenv("WILDLLAMA_IMAGES_DIR", service.root)
	restarted := &Server{}
	t.Cleanup(restarted.stopImageJobs)
	loaded := restarted.imageService()
	response := imageTestHTTP(mustImageRouter(t, restarted), "POST", uri, body, "image/png")
	require.Equal(t, 200, response.Code, response.Body.String())
	got, replayed := imageUploadedResponse(t, response)
	require.True(t, replayed)
	require.Equal(t, asset, got)
	require.Len(t, loaded.assets, 1)
	stored, err := os.ReadFile(asset.Path)
	require.NoError(t, err)
	require.Equal(t, body, stored)
	directories, err := os.ReadDir(filepath.Join(service.config.OutputDir, "inputs"))
	require.NoError(t, err)
	require.Len(t, directories, 1, "replay must not create another input directory")
}

func TestImageUploadIdempotencyRejectsPayloadAndScopeConflicts(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	body := imageUploadFixture(t, 220)
	response := imageTestHTTP(router, "POST", imageUploadURI("same-key", "chat-a"), body, "image/png")
	require.Equal(t, 201, response.Code)
	asset, _ := imageUploadedResponse(t, response)
	for _, item := range []struct {
		name, session, kind string
		body                []byte
	}{
		{"different bytes", "chat-a", "image/png", imageUploadFixture(t, 20)},
		{"different media type", "chat-a", "image/jpeg", body},
		{"different chat", "chat-b", "image/png", body},
		{"missing owner", "", "image/png", body},
	} {
		t.Run(item.name, func(t *testing.T) {
			response := imageTestHTTP(router, "POST", imageUploadURI("same-key", item.session), item.body, item.kind)
			require.Equal(t, 409, response.Code, response.Body.String())
			require.NotContains(t, response.Body.String(), asset.AssetID)
		})
	}
	require.Len(t, service.assets, 1)
	stored, err := os.ReadFile(asset.Path)
	require.NoError(t, err)
	require.Equal(t, body, stored, "conflicts must never replace the uploaded original")
	changed := imageUploadFixture(t, 40)
	require.NoError(t, os.WriteFile(asset.Path, changed, 0600))
	response = imageTestHTTP(router, "POST", imageUploadURI("same-key", "chat-a"), body, "image/png")
	require.Equal(t, 410, response.Code, "a changed original must not be returned as the original upload")
	stored, err = os.ReadFile(asset.Path)
	require.NoError(t, err)
	require.Equal(t, changed, stored)
}

func TestImageUploadIdempotencyConcurrentRetriesPublishOnce(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	body := imageUploadFixture(t, 220)
	const clients = 8
	responses := make([]*httptest.ResponseRecorder, clients)
	var group sync.WaitGroup
	for i := range responses {
		group.Add(1)
		go func() {
			defer group.Done()
			responses[i] = imageTestHTTP(router, "POST", imageUploadURI("concurrent-key", "chat-a"), body, "image/png")
		}()
	}
	group.Wait()
	created, id := 0, ""
	for _, response := range responses {
		require.Contains(t, []int{200, 201}, response.Code, response.Body.String())
		if response.Code == 201 {
			created++
		}
		asset, _ := imageUploadedResponse(t, response)
		if id == "" {
			id = asset.AssetID
		}
		require.Equal(t, id, asset.AssetID)
	}
	require.Equal(t, 1, created)
	require.Len(t, service.assets, 1)
	directories, err := os.ReadDir(filepath.Join(service.config.OutputDir, "inputs"))
	require.NoError(t, err)
	require.Len(t, directories, 1)
}

func TestImageUploadIdempotencyBoundsAndLegacyUploads(t *testing.T) {
	server, service, router, _, _ := newImageTestService(t)
	require.Equal(t, true, server.protocolCapabilities()["features"].(gin.H)["image_upload_idempotency"])
	body := imageUploadFixture(t, 220)
	for _, uri := range []string{
		imageUploadURI("../path", "chat-a"), imageUploadURI(strings.Repeat("x", 129), "chat-a"),
		imageUploadURI("key", strings.Repeat("s", 257)), imageUploadURI("key", "bad\x00session"),
		"/api/images/assets?request_id=one&request_id=two",
	} {
		response := imageTestHTTP(router, "POST", uri, body, "image/png")
		require.Equal(t, 400, response.Code, response.Body.String())
	}
	require.Empty(t, service.assets)
	for range 2 {
		response := imageTestHTTP(router, "POST", "/api/images/assets", body, "image/png")
		require.Equal(t, 201, response.Code, response.Body.String())
	}
	require.Len(t, service.assets, 2, "uploads without request_id retain their existing behavior")
}

type imageBlockedUpload struct {
	entered chan struct{}
	release <-chan struct{}
	source  io.Reader
	once    sync.Once
}

func (r *imageBlockedUpload) Read(p []byte) (int, error) {
	r.once.Do(func() { close(r.entered); <-r.release })
	return r.source.Read(p)
}

func TestImageUploadStreamingDoesNotBlockJobStatusOrCancellation(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	service.workerStarted = true
	job, err := service.enqueue(imageLifecycleJob(service, ""))
	require.NoError(t, err)
	release := make(chan struct{})
	var once sync.Once
	body := &imageBlockedUpload{entered: make(chan struct{}), release: release, source: bytes.NewReader(imageUploadFixture(t, 220))}
	request := httptest.NewRequest("POST", imageUploadURI("slow-upload", "chat-a"), body)
	request.Header.Set("Content-Type", "image/png")
	response := httptest.NewRecorder()
	done := make(chan struct{})
	go func() { defer close(done); router.ServeHTTP(response, request) }()
	t.Cleanup(func() { once.Do(func() { close(release) }); <-done })
	select {
	case <-body.entered:
	case <-time.After(time.Second):
		t.Fatal("upload did not begin reading")
	}
	status := make(chan *httptest.ResponseRecorder, 1)
	go func() { status <- imageTestHTTP(router, "GET", "/api/images/jobs", nil, "") }()
	select {
	case got := <-status:
		require.Equal(t, 200, got.Code)
	case <-time.After(time.Second):
		t.Fatal("streaming upload blocked job status")
	}
	cancel := make(chan *httptest.ResponseRecorder, 1)
	go func() { cancel <- imageTestHTTP(router, "POST", "/api/images/jobs/"+job.ID+"/cancel", nil, "") }()
	select {
	case got := <-cancel:
		require.Equal(t, 200, got.Code)
	case <-time.After(time.Second):
		t.Fatal("streaming upload blocked cancellation")
	}
	once.Do(func() { close(release) })
	<-done
	require.Equal(t, 201, response.Code, response.Body.String())
}
