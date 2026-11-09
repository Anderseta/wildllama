package server

import (
	"context"
	"crypto/md5"
	"errors"
	"fmt"
	"hash"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

var blobUploadManager sync.Map

type blobUpload struct {
	manifest.Layer

	Total     int64
	Completed atomic.Int64

	Parts []blobUploadPart

	nextURL chan *url.URL

	context.CancelFunc

	file *os.File

	context     context.Context
	done        chan struct{}
	ready       chan struct{}
	mounted     bool
	err         error // published by closing done
	references  atomic.Int32
	referenceMu sync.Mutex
}

const (
	numUploadParts          = 16
	minUploadPartSize int64 = 100 * format.MegaByte
	maxUploadPartSize int64 = 1000 * format.MegaByte
)

func (b *blobUpload) Prepare(ctx context.Context, requestURL *url.URL, opts *registryOptions) error {
	p, err := manifest.BlobsPath(b.Digest)
	if err != nil {
		return err
	}

	if b.From != "" {
		values := requestURL.Query()
		values.Add("mount", b.Digest)
		values.Add("from", model.ParseName(b.From).DisplayNamespaceModel())
		requestURL.RawQuery = values.Encode()
	}

	resp, err := makeRequestWithRetry(ctx, http.MethodPost, requestURL, nil, nil, opts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	location := resp.Header.Get("Docker-Upload-Location")
	if location == "" {
		location = resp.Header.Get("Location")
	}

	fi, err := os.Stat(p)
	if err != nil {
		return err
	}

	b.Total = fi.Size()

	// http.StatusCreated indicates a blob has been mounted
	// ref: https://distribution.github.io/distribution/spec/api/#cross-repository-blob-mount
	if resp.StatusCode == http.StatusCreated {
		b.Completed.Store(b.Total)
		b.mounted = true
		return nil
	}

	size := b.Total / numUploadParts
	switch {
	case size < minUploadPartSize:
		size = minUploadPartSize
	case size > maxUploadPartSize:
		size = maxUploadPartSize
	}

	var offset int64
	for offset < fi.Size() {
		if offset+size > fi.Size() {
			size = fi.Size() - offset
		}

		// set part.N to the current number of parts
		b.Parts = append(b.Parts, blobUploadPart{N: len(b.Parts), Offset: offset, Size: size})
		offset += size
	}

	if len(b.Parts) > 0 {
		slog.Info(fmt.Sprintf("uploading %s in %d %s part(s)", b.Digest[7:19], len(b.Parts), format.HumanBytes(b.Parts[0].Size)))
	}

	nextURL, err := url.Parse(location)
	if err != nil {
		return err
	}

	b.nextURL = make(chan *url.URL, 1)
	b.nextURL <- requestURL.ResolveReference(nextURL)
	return nil
}

// run uploads prepared parts. The shared transfer publishes its result through done.
func (b *blobUpload) run(ctx context.Context, opts *registryOptions) error {
	if b.mounted {
		return nil
	}

	p, err := manifest.BlobsPath(b.Digest)
	if err != nil {
		return err
	}

	b.file, err = os.Open(p)
	if err != nil {
		return err
	}
	defer b.file.Close()

	g, inner := errgroup.WithContext(ctx)
	g.SetLimit(numUploadParts)
	for i := range b.Parts {
		part := &b.Parts[i]
		select {
		case <-inner.Done():
		case requestURL := <-b.nextURL:
			g.Go(func() error {
				partOpts := *opts // auth retries must not race other parts' tokens
				var err error
				for try := range maxRetries {
					err = b.uploadPart(inner, http.MethodPatch, requestURL, part, &partOpts)
					switch {
					case errors.Is(err, context.Canceled):
						return err
					case errors.Is(err, errMaxRetriesExceeded):
						return err
					case err != nil:
						sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
						slog.Info(fmt.Sprintf("%s part %d attempt %d failed: %v, retrying in %s", b.Digest[7:19], part.N, try, err, sleep))
						if err := waitUploadRetry(inner, sleep); err != nil {
							return err
						}
						continue
					}

					return nil
				}

				return fmt.Errorf("%w: %w", errMaxRetriesExceeded, err)
			})
		}
	}

	if err := g.Wait(); err != nil {
		return err
	}

	var requestURL *url.URL
	select {
	case <-ctx.Done():
		return ctx.Err()
	case requestURL = <-b.nextURL:
	}

	// calculate md5 checksum and add it to the commit request
	md5sum := md5.New()
	for _, part := range b.Parts {
		md5sum.Write(part.Sum(nil))
	}

	values := requestURL.Query()
	values.Add("digest", b.Digest)
	values.Add("etag", fmt.Sprintf("%x-%d", md5sum.Sum(nil), len(b.Parts)))
	requestURL.RawQuery = values.Encode()

	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", "0")

	for try := range maxRetries {
		var resp *http.Response
		resp, err = makeRequestWithRetry(ctx, http.MethodPut, requestURL, headers, nil, opts)
		if errors.Is(err, context.Canceled) {
			break
		} else if err != nil {
			sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
			slog.Info(fmt.Sprintf("%s complete upload attempt %d failed: %v, retrying in %s", b.Digest[7:19], try, err, sleep))
			if err := waitUploadRetry(ctx, sleep); err != nil {
				return err
			}
			continue
		}
		defer resp.Body.Close()
		break
	}

	return err
}

func waitUploadRetry(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (b *blobUpload) uploadPart(ctx context.Context, method string, requestURL *url.URL, part *blobUploadPart, opts *registryOptions) error {
	headers := make(http.Header)
	headers.Set("Content-Type", "application/octet-stream")
	headers.Set("Content-Length", strconv.FormatInt(part.Size, 10))

	if method == http.MethodPatch {
		headers.Set("X-Redirect-Uploads", "1")
		headers.Set("Content-Range", fmt.Sprintf("%d-%d", part.Offset, part.Offset+part.Size-1))
	}

	sr := io.NewSectionReader(b.file, part.Offset, part.Size)

	md5sum := md5.New()
	w := &progressWriter{blobUpload: b}

	resp, err := makeRequest(ctx, method, requestURL, headers, io.TeeReader(sr, io.MultiWriter(w, md5sum)), opts)
	if err != nil {
		w.Rollback()
		return err
	}
	defer resp.Body.Close()

	location := resp.Header.Get("Docker-Upload-Location")
	if location == "" {
		location = resp.Header.Get("Location")
	}

	nextURL, err := url.Parse(location)
	if err != nil {
		w.Rollback()
		return err
	}
	nextURL = requestURL.ResolveReference(nextURL)

	switch {
	case resp.StatusCode == http.StatusTemporaryRedirect:
		w.Rollback()
		select {
		case b.nextURL <- nextURL:
		case <-ctx.Done():
			return ctx.Err()
		}

		redirectURL, err := resp.Location()
		if err != nil {
			return err
		}

		// retry uploading to the redirect URL
		for try := range maxRetries {
			err = b.uploadPart(ctx, http.MethodPut, redirectURL, part, &registryOptions{})
			switch {
			case errors.Is(err, context.Canceled):
				return err
			case errors.Is(err, errMaxRetriesExceeded):
				return err
			case err != nil:
				sleep := time.Second * time.Duration(math.Pow(2, float64(try)))
				slog.Info(fmt.Sprintf("%s part %d attempt %d failed: %v, retrying in %s", b.Digest[7:19], part.N, try, err, sleep))
				if err := waitUploadRetry(ctx, sleep); err != nil {
					return err
				}
				continue
			}

			return nil
		}

		return fmt.Errorf("%w: %w", errMaxRetriesExceeded, err)

	case resp.StatusCode == http.StatusUnauthorized:
		w.Rollback()
		challenge := parseRegistryChallenge(resp.Header.Get("www-authenticate"))
		token, err := getAuthorizationToken(ctx, challenge, requestURL.Host)
		if err != nil {
			return err
		}

		opts.Token = token
		fallthrough
	case resp.StatusCode >= http.StatusBadRequest:
		w.Rollback()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return err
		}

		return fmt.Errorf("http status %s: %s", resp.Status, body)
	}

	if method == http.MethodPatch {
		select {
		case b.nextURL <- nextURL:
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	part.Hash = md5sum
	return nil
}

func (b *blobUpload) acquire() bool {
	b.referenceMu.Lock()
	defer b.referenceMu.Unlock()
	if b.context.Err() != nil {
		return false
	}
	b.references.Add(1)
	return true
}

func (b *blobUpload) release() {
	b.referenceMu.Lock()
	defer b.referenceMu.Unlock()
	if b.references.Add(-1) == 0 {
		b.CancelFunc()
	}
}

func (b *blobUpload) Wait(ctx context.Context, fn func(api.ProgressResponse)) error {
	finish := func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if b.err == nil {
			fn(api.ProgressResponse{Status: fmt.Sprintf("pushing %s", b.Digest[7:19]),
				Digest: b.Digest, Total: b.Total, Completed: b.Completed.Load()})
		}
		return b.err
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-b.done:
		return finish()
	case <-b.ready:
	}

	ticker := time.NewTicker(60 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-b.done:
			return finish()
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}

		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pushing %s", b.Digest[7:19]),
			Digest:    b.Digest,
			Total:     b.Total,
			Completed: b.Completed.Load(),
		})

	}
}

type blobUploadPart struct {
	// N is the part number
	N      int
	Offset int64
	Size   int64
	hash.Hash
}

type progressWriter struct {
	written int64
	*blobUpload
}

func (p *progressWriter) Write(b []byte) (n int, err error) {
	n = len(b)
	p.written += int64(n)
	p.Completed.Add(int64(n))
	return n, nil
}

func (p *progressWriter) Rollback() {
	p.Completed.Add(-p.written)
	p.written = 0
}

func uploadBlob(ctx context.Context, n model.Name, layer manifest.Layer, opts *registryOptions, fn func(api.ProgressResponse)) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	requestURL := n.BaseURL()
	requestURL = requestURL.JoinPath("v2", n.DisplayNamespaceModel(), "blobs", layer.Digest)

	resp, err := makeRequestWithRetry(ctx, http.MethodHead, requestURL, nil, nil, opts)
	switch {
	case errors.Is(err, os.ErrNotExist):
	case err != nil:
		return err
	default:
		defer resp.Body.Close()
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pushing %s", layer.Digest[7:19]),
			Digest:    layer.Digest,
			Total:     layer.Size,
			Completed: layer.Size,
		})

		return nil
	}

	// A digest alone cannot identify an upload: different repositories/registries
	// each need their own session, even when their model weights are identical.
	key := requestURL.String()
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		shared, cancel := context.WithCancel(context.Background())
		candidate := &blobUpload{Layer: layer, context: shared, CancelFunc: cancel,
			done: make(chan struct{}), ready: make(chan struct{})}
		candidate.acquire() // own the first consumer before publishing
		data, loaded := blobUploadManager.LoadOrStore(key, candidate)
		upload := data.(*blobUpload)
		if loaded {
			cancel()
			if !upload.acquire() {
				select {
				case <-ctx.Done():
					return ctx.Err()
				case <-upload.done:
					continue
				}
			}
		} else {
			// Isolate mutable authentication state from the caller and other parts.
			sharedOpts := *opts
			go func() {
				defer close(upload.done)
				defer blobUploadManager.CompareAndDelete(key, upload)
				defer cancel()
				start := n.BaseURL().JoinPath("v2", n.DisplayNamespaceModel(), "blobs/uploads/")
				if upload.err = upload.Prepare(shared, start, &sharedOpts); upload.err != nil {
					return
				}
				close(upload.ready)
				upload.err = upload.run(shared, &sharedOpts)
			}()
		}
		defer upload.release()
		return upload.Wait(ctx, fn)
	}
}
