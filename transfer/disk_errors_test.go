package transfer

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type failedTransferWriter struct {
	err   error
	short bool
}

func (w failedTransferWriter) Write(p []byte) (int, error) {
	if w.short {
		return min(2, len(p)), nil
	}
	return min(2, len(p)), w.err
}

func TestTransferCopyReportsActualShortAndFailedWrites(t *testing.T) {
	for _, short := range []bool{false, true} {
		t.Run(map[bool]string{false: "error", true: "short"}[short], func(t *testing.T) {
			failure := errors.New("disk full fixture")
			tracker := newProgressTracker(6, func(int64, int64) {})
			downloader := &downloader{progress: tracker, speeds: &speedTracker{}, stallTimeout: time.Second}
			ctx, cancel := context.WithCancelCause(t.Context())
			defer cancel(nil)
			hash := sha256.New()
			n, err := downloader.copy(ctx, cancel, failedTransferWriter{err: failure, short: short}, strings.NewReader("abcdef"), hash)
			require.Equal(t, int64(2), n)
			require.Equal(t, int64(2), tracker.completed.Load())
			if short {
				require.ErrorIs(t, err, io.ErrShortWrite)
			} else {
				require.ErrorIs(t, err, failure)
			}
			expected := sha256.Sum256([]byte("ab"))
			require.Equal(t, expected[:], hash.Sum(nil))
		})
	}
}

type unreadTransferBody struct {
	reads  int
	closed bool
}

func (b *unreadTransferBody) Read([]byte) (int, error) {
	b.reads++
	return 0, errors.New("discovery must not consume blob body")
}
func (b *unreadTransferBody) Close() error { b.closed = true; return nil }

type diskTestTransport func(*http.Request) (*http.Response, error)

func (f diskTestTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func TestTransferDiscoveryDoesNotDownloadSuccessfulBlobBody(t *testing.T) {
	body := &unreadTransferBody{}
	client := &http.Client{Transport: diskTestTransport(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: body, Header: make(http.Header)}, nil
	})}
	downloader := &downloader{client: client, baseURL: "https://fixture.invalid", token: ""}
	address, err := downloader.resolve(t.Context(), "https://fixture.invalid/blob")
	require.NoError(t, err)
	require.Equal(t, "https://fixture.invalid/blob", address.String())
	require.Zero(t, body.reads)
	require.True(t, body.closed)
}

func TestTransferPermanentDiskErrorDoesNotRetryOrRemovePartial(t *testing.T) {
	directory := t.TempDir()
	digest := "sha256:" + strings.Repeat("a", 64)
	failure := errors.New("space guard fixture")
	attempts := 0
	client := &http.Client{Transport: diskTestTransport(func(request *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(bytes.NewReader([]byte("abc"))), Header: make(http.Header)}, nil
	})}
	partial := filepath.Join(directory, digestToPath(digest)+".tmp")
	require.NoError(t, os.WriteFile(partial, []byte("partial"), 0600))
	downloader := &downloader{client: client, baseURL: "https://fixture.invalid", destDir: directory, repository: "library/test", progress: newProgressTracker(3, func(int64, int64) {}), speeds: &speedTracker{}, reserveSpace: func(context.Context, string, int64) (DiskReservation, error) { attempts++; return nil, failure }, permanentError: func(err error) bool { return errors.Is(err, failure) }}
	err := downloader.download(t.Context(), Blob{Digest: digest, Size: 3})
	require.ErrorIs(t, err, failure)
	require.Equal(t, 1, attempts)
	contents, readErr := os.ReadFile(partial)
	require.NoError(t, readErr)
	require.Equal(t, "partial", string(contents))
}
