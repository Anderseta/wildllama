package server

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type imageCancelOnEOF struct {
	reader io.Reader
	cancel context.CancelFunc
}

func (r *imageCancelOnEOF) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if err == io.EOF {
		r.cancel()
	}
	return n, err
}

func TestImageUploadCancellationBeforeValidationDoesNotPublish(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	body := imageUploadFixture(t, 220)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	uri := imageUploadURI("cancel-at-upload-eof", "chat-a")
	reader := &imageCancelOnEOF{reader: bytes.NewReader(body), cancel: cancel}
	request := httptest.NewRequest(http.MethodPost, uri, reader).WithContext(ctx)
	request.Header.Set("Content-Type", "image/png")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	require.ErrorIs(t, ctx.Err(), context.Canceled)
	require.Equal(t, 499, response.Code, response.Body.String())
	require.Empty(t, service.assets, "a cancelled upload must not publish an orphaned input")
	dirs, err := os.ReadDir(filepath.Join(service.config.OutputDir, "inputs"))
	require.NoError(t, err)
	require.Empty(t, dirs, "the incomplete upload's own temporary directory must be removed")
	// Cancellation releases the serialized decoder slot and does not reserve
	// the idempotency key; the ordinary retry can publish the exact input.
	retry := imageTestHTTP(router, http.MethodPost, uri, body, "image/png")
	require.Equal(t, http.StatusCreated, retry.Code, retry.Body.String())
	asset, replayed := imageUploadedResponse(t, retry)
	require.False(t, replayed)
	require.Len(t, service.assets, 1)
	stored, err := os.ReadFile(asset.Path)
	require.NoError(t, err)
	require.Equal(t, body, stored)
}

func TestImageUploadCancelledReplayPreservesCommittedOriginal(t *testing.T) {
	_, service, router, _, _ := newImageTestService(t)
	body := imageUploadFixture(t, 140)
	uri := imageUploadURI("cancelled-replay", "chat-a")
	created := imageTestHTTP(router, http.MethodPost, uri, body, "image/png")
	require.Equal(t, http.StatusCreated, created.Code)
	asset, _ := imageUploadedResponse(t, created)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	reader := &imageCancelOnEOF{reader: bytes.NewReader(body), cancel: cancel}
	request := httptest.NewRequest(http.MethodPost, uri, reader).WithContext(ctx)
	request.Header.Set("Content-Type", "image/png")
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	require.Equal(t, 499, response.Code, response.Body.String())
	require.Len(t, service.assets, 1)
	stored, err := os.ReadFile(asset.Path)
	require.NoError(t, err)
	require.Equal(t, body, stored, "cancelling a replay must preserve the committed original")
	retry := imageTestHTTP(router, http.MethodPost, uri, body, "image/png")
	require.Equal(t, http.StatusOK, retry.Code)
	got, replayed := imageUploadedResponse(t, retry)
	require.True(t, replayed)
	require.Equal(t, asset, got)
}

type imageUploadReadCounter struct {
	reader io.Reader
	reads  int
}

func (r *imageUploadReadCounter) Read(p []byte) (int, error) {
	r.reads++
	return r.reader.Read(p)
}

func TestImageUploadCancelledBeforeDecoderAdmissionDoesNotRead(t *testing.T) {
	for _, waiting := range []bool{false, true} {
		t.Run(map[bool]string{false: "before-request", true: "waiting-for-decoder"}[waiting], func(t *testing.T) {
			_, service, router, _, _ := newImageTestService(t)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if waiting {
				service.uploadSlot <- struct{}{}
				defer func() { <-service.uploadSlot }()
			} else {
				cancel()
			}
			reader := &imageUploadReadCounter{reader: bytes.NewReader(imageUploadFixture(t, 110))}
			request := httptest.NewRequest(http.MethodPost, imageUploadURI("cancel-before-decode", "chat-a"), reader).WithContext(ctx)
			request.Header.Set("Content-Type", "image/png")
			response := httptest.NewRecorder()
			done := make(chan struct{})
			go func() { router.ServeHTTP(response, request); close(done) }()
			if waiting {
				cancel()
			}
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Fatal("cancelled upload did not release the decoder admission wait")
			}
			require.Equal(t, 499, response.Code, response.Body.String())
			require.Zero(t, reader.reads)
			require.Empty(t, service.assets)
		})
	}
}
