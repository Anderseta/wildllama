package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
	"github.com/stretchr/testify/require"
)

func testModelDiskBudget(free *int64) *modelDiskBudget {
	return &modelDiskBudget{filesystem: func(string) (string, error) { return "fixture-volume", nil }, check: func(_ context.Context, _ string, remaining int64) error {
		if remaining > *free {
			return imagegen.ErrInsufficientDiskSpace
		}
		return nil
	}}
}

func TestModelDiskBudgetReservesAggregateAndReleasesWrittenBytes(t *testing.T) {
	t.Setenv("WILDLLAMA_MIN_FREE_DISK_MIB", "1")
	free := int64(100)
	budget := testModelDiskBudget(&free)
	first, err := budget.reserve(t.Context(), "first", 60)
	require.NoError(t, err)
	defer first.close()
	_, err = budget.reserve(t.Context(), "second", 50)
	require.ErrorIs(t, err, imagegen.ErrInsufficientDiskSpace)
	first.consume(40)
	free -= 40
	second, err := budget.reserve(t.Context(), "second", 40)
	require.NoError(t, err)
	second.close()
	second.close()
	require.Len(t, budget.active, 1)
	first.close()
	require.Empty(t, budget.active)
}

func TestModelDiskWriterDoesNotDoubleReserveRetryRanges(t *testing.T) {
	t.Setenv("WILDLLAMA_MIN_FREE_DISK_MIB", "1")
	free := int64(100)
	budget := testModelDiskBudget(&free)
	reservation, err := budget.reserve(t.Context(), "fixture", 10)
	require.NoError(t, err)
	defer reservation.close()
	part := &blobDownloadPart{}
	first := &modelDiskWriter{writer: io.Discard, reservation: reservation, part: part}
	_, err = first.Write([]byte("abc"))
	require.NoError(t, err)
	require.Equal(t, int64(7), reservation.remaining.Load())
	retry := &modelDiskWriter{writer: io.Discard, reservation: reservation, part: part}
	_, err = retry.Write([]byte("abcd"))
	require.NoError(t, err)
	require.Equal(t, int64(6), reservation.remaining.Load())
}

func TestModelDiskBudgetChecksUnknownLengthBeforePublishingReadBytes(t *testing.T) {
	t.Setenv("WILDLLAMA_MIN_FREE_DISK_MIB", "1")
	free := int64(5)
	budget := testModelDiskBudget(&free)
	reservation, err := budget.reserve(t.Context(), "fixture", 0)
	require.NoError(t, err)
	defer reservation.close()
	reader := &modelDiskReader{source: bytes.NewBufferString("abcdefgh"), reservation: reservation, unknown: true}
	buffer := make([]byte, 4)
	n, err := reader.Read(buffer)
	require.NoError(t, err)
	require.Equal(t, 4, n)
	free -= 4
	n, err = reader.Read(buffer)
	require.Zero(t, n)
	require.ErrorIs(t, err, imagegen.ErrInsufficientDiskSpace)
}

func TestModelDiskBudgetCanBeDisabled(t *testing.T) {
	t.Setenv("WILDLLAMA_MIN_FREE_DISK_MIB", "0")
	budget := &modelDiskBudget{}
	reservation, err := budget.reserve(t.Context(), "not-a-path", 100)
	require.NoError(t, err)
	require.Nil(t, reservation)
}

func TestCreateBlobDiskPreflightRejectsBeforeWriting(t *testing.T) {
	t.Setenv("WILDLLAMA_MIN_FREE_DISK_MIB", "1")
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)
	previous := modelTransfersDisk
	t.Cleanup(func() { modelTransfersDisk = previous })
	free := int64(0)
	modelTransfersDisk = *testModelDiskBudget(&free)
	payload := []byte("model-fixture")
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(payload))
	server := &Server{}
	router := gin.New()
	router.POST("/api/blobs/:digest", server.CreateBlobHandler)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/api/blobs/"+digest, bytes.NewReader(payload)))
	require.Equal(t, http.StatusInsufficientStorage, response.Code, response.Body.String())
	entries, err := os.ReadDir(filepath.Join(models, "blobs"))
	require.NoError(t, err)
	require.Empty(t, entries)
}
