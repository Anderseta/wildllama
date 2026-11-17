package transfer

import (
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/sync/semaphore"
)

var (
	errStalled = errors.New("download stalled")
	errSlow    = errors.New("download too slow")
)

type downloader struct {
	client         *http.Client
	baseURL        string
	destDir        string
	repository     string // Repository path for blob URLs (e.g., "library/model")
	tokenMu        sync.RWMutex
	token          string
	getToken       func(context.Context, AuthChallenge) (string, error)
	userAgent      string
	stallTimeout   time.Duration
	progress       *progressTracker
	speeds         *speedTracker
	logger         *slog.Logger
	allowPrivate   bool
	reserveSpace   func(context.Context, string, int64) (DiskReservation, error)
	permanentError func(error) bool
	// bodySem caps the number of simultaneous body-bearing transfers so a
	// modest home downlink isn't saturated. Always set by download(); nil
	// only when tests build downloader directly (in which case holdBody is
	// a no-op).
	bodySem *semaphore.Weighted
}

// authToken returns the current bearer token. Safe to call concurrently with
// refreshToken.
func (d *downloader) authToken() string {
	d.tokenMu.RLock()
	defer d.tokenMu.RUnlock()
	return d.token
}

// refreshToken coalesces token fetches so concurrent 401s don't all hit the
// auth server. prev is the token the caller used in the request that got
// rejected: if the stored token has already moved past prev, another
// goroutine has refreshed and we just observe its result; otherwise the
// caller holds the lock and performs the fetch.
func (d *downloader) refreshToken(ctx context.Context, ch AuthChallenge, prev string) error {
	d.tokenMu.Lock()
	defer d.tokenMu.Unlock()
	if d.token != prev {
		return nil
	}
	if d.getToken == nil {
		return errors.New("no token refresh callback")
	}
	t, err := d.getToken(ctx, ch)
	if err != nil {
		return err
	}
	d.token = t
	return nil
}

// holdBody acquires a body-transfer slot. The returned release must be
// called exactly once after the body-bearing request completes (defer is fine).
func (d *downloader) holdBody(ctx context.Context) (func(), error) {
	if d.bodySem == nil {
		return func() {}, nil
	}
	if err := d.bodySem.Acquire(ctx, 1); err != nil {
		return nil, err
	}
	return func() { d.bodySem.Release(1) }, nil
}

func download(ctx context.Context, opts DownloadOptions) error {
	if len(opts.Blobs) == 0 {
		return nil
	}

	var blobs []Blob
	var total, alreadyCompleted, remaining int64
	seen := make(map[string]int64, len(opts.Blobs))
	for _, blob := range opts.Blobs {
		if size, ok := seen[blob.Digest]; ok {
			if size != blob.Size {
				return fmt.Errorf("conflicting sizes for blob %s", blob.Digest)
			}
			continue
		}
		if blob.Size < 0 || blob.Size > (1<<62)-total {
			return errors.New("invalid model download size")
		}
		seen[blob.Digest] = blob.Size
		total += blob.Size
		path := filepath.Join(opts.DestDir, digestToPath(blob.Digest))
		if fi, err := os.Stat(path); err == nil && fi.Mode().IsRegular() && fi.Size() == blob.Size {
			alreadyCompleted += blob.Size
			continue
		}
		blobs = append(blobs, blob)
		if opts.CheckSpace != nil {
			remaining += blob.Size
			if blob.Size >= resumeThreshold {
				if fi, err := os.Stat(path + ".tmp"); err == nil && fi.Mode().IsRegular() && fi.Size() <= blob.Size {
					remaining -= fi.Size()
				}
			}
		}
	}
	if remaining > 0 {
		if err := opts.CheckSpace(ctx, opts.DestDir, remaining); err != nil {
			return err
		}
	}
	progress := newProgressTracker(total, opts.Progress)
	progress.add(alreadyCompleted)
	if len(blobs) == 0 {
		return nil
	}

	d := &downloader{
		client:         cmp.Or(opts.Client, checkedClient(opts.BaseURL, opts.AllowPrivateHosts)),
		baseURL:        opts.BaseURL,
		destDir:        opts.DestDir,
		repository:     cmp.Or(opts.Repository, "library/_"),
		token:          opts.Token,
		getToken:       opts.GetToken,
		userAgent:      cmp.Or(opts.UserAgent, defaultUserAgent),
		stallTimeout:   cmp.Or(opts.StallTimeout, defaultStallTimeout),
		progress:       progress,
		speeds:         &speedTracker{},
		logger:         opts.Logger,
		allowPrivate:   opts.AllowPrivateHosts,
		reserveSpace:   opts.ReserveSpace,
		permanentError: opts.PermanentError,
	}
	// 0 or negative serializes; never unbounded.
	d.bodySem = semaphore.NewWeighted(int64(max(1, opts.BodyConcurrency)))

	concurrency := max(1, cmp.Or(opts.Concurrency, DefaultDownloadConcurrency))
	sem := semaphore.NewWeighted(int64(concurrency))

	start := time.Now()
	g, ctx := errgroup.WithContext(ctx)
	for _, blob := range blobs {
		g.Go(func() error {
			if err := sem.Acquire(ctx, 1); err != nil {
				return err
			}
			defer sem.Release(1)
			return d.download(ctx, blob)
		})
	}
	err := g.Wait()
	elapsed := time.Since(start)
	done := d.progress.completed.Load() - alreadyCompleted
	mbps := float64(done) / 1e6 / max(0.001, elapsed.Seconds())
	slog.Debug("download summary",
		"blobs", len(blobs),
		"bytes", done,
		"duration", elapsed.Round(time.Millisecond),
		"mb_per_sec", fmt.Sprintf("%.1f", mbps),
		"max_transfers", max(1, opts.BodyConcurrency),
	)
	return err
}

func (d *downloader) download(ctx context.Context, blob Blob) error {
	var lastErr error
	var slowRetries, stallRetries, retries int
	attempt := 0

	for attempt < maxRetries {
		// Keyed to every retry, not just the ones that count against the limit,
		// so repeated stalls cannot spin without pausing.
		if retries > 0 {
			if err := backoff(ctx, retries, time.Second<<uint(min(retries, maxRetries)-1)); err != nil {
				return err
			}
		}

		start := time.Now()
		n, err := d.downloadOnce(ctx, blob)
		if err == nil {
			// Skip speed recording for tiny blobs — their transfer time is
			// dominated by HTTP overhead, not throughput, and would pollute
			// the median used for stall detection.
			if n >= smallBlobSpeedThreshold {
				if s := time.Since(start).Seconds(); s > 0 {
					d.speeds.record(float64(n) / s)
				}
			}
			return nil
		}

		d.progress.add(-n) // rollback
		if errors.Is(err, ErrInvalidDownloadRange) {
			return err
		}
		if d.permanentError != nil && d.permanentError(err) {
			return err
		}

		// Preserve partial .tmp files for large blobs to enable resume
		if blob.Size < resumeThreshold {
			dest := filepath.Join(d.destDir, digestToPath(blob.Digest))
			os.Remove(dest + ".tmp")
		}

		retries++

		switch {
		case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
			return err
		case errors.Is(err, errStalled):
			if stallRetries++; stallRetries >= maxTransientRetries {
				attempt++
			}
		case errors.Is(err, errSlow):
			if slowRetries++; slowRetries >= maxTransientRetries {
				attempt++
			}
		default:
			attempt++
		}
		lastErr = err
	}
	return fmt.Errorf("%w: %w", errMaxRetriesExceeded, lastErr)
}

func (d *downloader) downloadOnce(ctx context.Context, blob Blob) (int64, error) {
	if d.logger != nil {
		d.logger.Debug("downloading blob", "digest", blob.Digest, "size", blob.Size)
	}

	// Hold a body slot for the duration of the GET — released when the body
	// has been read and the response closed.
	release, err := d.holdBody(ctx)
	if err != nil {
		return 0, err
	}
	defer release()

	var existingSize int64
	if blob.Size >= resumeThreshold {
		dest := filepath.Join(d.destDir, digestToPath(blob.Digest))
		tmp := dest + ".tmp"
		fi, err := os.Stat(tmp)
		switch {
		case errors.Is(err, os.ErrNotExist):
		case err != nil:
			return 0, err
		case fi.Size() == blob.Size:
			// A previous process may have finished writing just before publication.
			valid, err := verifyCompletePartial(ctx, tmp, blob)
			if err != nil {
				return 0, err
			}
			if valid {
				if err := os.Rename(tmp, dest); err != nil {
					return 0, err
				}
				d.progress.add(blob.Size)
				return 0, nil // no network bytes for the speed estimate
			}
			if err := os.Remove(tmp); err != nil {
				return 0, err
			}
		case fi.Size() < blob.Size:
			existingSize = fi.Size()
		}
	}

	baseURL, _ := url.Parse(d.baseURL)
	u, err := d.resolve(ctx, fmt.Sprintf("%s/v2/%s/blobs/%s", d.baseURL, d.repository, blob.Digest))
	if err != nil {
		return 0, err
	}

	// The body read is cancelled through this context, so the stall watchdog in
	// copy can actually interrupt a request that has stopped delivering bytes.
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	req.Header.Set("User-Agent", d.userAgent)
	// Add auth only for same-host (not CDN)
	if u.Host == baseURL.Host {
		if t := d.authToken(); t != "" {
			req.Header.Set("Authorization", "Bearer "+t)
		}
	}
	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		// Full response — reset any partial state
		existingSize = 0
	}
	if err := ValidateDownloadResponse(resp, existingSize, blob.Size-existingSize, blob.Size); err != nil {
		return 0, err
	}

	n, err := d.save(ctx, cancel, blob, resp.Body, existingSize)
	if err != nil {
		return n, err // bytes credited to progress, to roll back on failure
	}
	return n - existingSize, nil
}

type partialFileReader struct {
	ctx  context.Context
	file *os.File
}

func (r partialFileReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.file.Read(p)
}

func verifyCompletePartial(ctx context.Context, path string, blob Blob) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, partialFileReader{ctx, f})
	if err != nil {
		return false, err
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	return n == blob.Size && fmt.Sprintf("sha256:%x", h.Sum(nil)) == blob.Digest, nil
}

func (d *downloader) save(ctx context.Context, cancel context.CancelCauseFunc, blob Blob, r io.Reader, existingSize int64) (int64, error) {
	dest := filepath.Join(d.destDir, digestToPath(blob.Digest))
	tmp := dest + ".tmp"
	var reservation DiskReservation
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return 0, err
	}

	h := sha256.New()

	var f *os.File
	var err error

	if d.reserveSpace != nil {
		reservation, err = d.reserveSpace(ctx, tmp, max(0, blob.Size-existingSize))
		if err != nil {
			return 0, err
		}
		if reservation != nil {
			defer reservation.Close()
		}
	}
	if existingSize > 0 {
		f, err = os.OpenFile(tmp, os.O_RDWR, 0)
		if err != nil {
			return 0, err
		}
		if _, err := io.CopyN(h, partialFileReader{ctx, f}, existingSize); err != nil {
			f.Close()
			return 0, err // a range response cannot replace a missing prefix
		}
		d.progress.add(existingSize)
	} else {
		f, err = os.Create(tmp)
		if err != nil {
			return 0, err
		}
		setSparse(f)
	}
	defer f.Close()

	var destination io.Writer = f
	if reservation != nil {
		destination = reservedDiskWriter{writer: f, reservation: reservation}
	}
	n, err := d.copy(ctx, cancel, destination, r, h)
	if err != nil {
		// Don't remove .tmp here — download() handles cleanup based on blob size
		return existingSize + n, err
	}
	if err := f.Close(); err != nil {
		return existingSize + n, err
	}

	if got := fmt.Sprintf("sha256:%x", h.Sum(nil)); got != blob.Digest {
		os.Remove(tmp)
		return existingSize + n, fmt.Errorf("digest mismatch")
	}
	totalWritten := existingSize + n
	if totalWritten != blob.Size {
		os.Remove(tmp)
		return totalWritten, fmt.Errorf("size mismatch")
	}
	return totalWritten, os.Rename(tmp, dest)
}

// copy streams src to dst, cancelling the caller's request context when the
// transfer stalls or falls far behind the median speed.
func (d *downloader) copy(ctx context.Context, cancel context.CancelCauseFunc, dst io.Writer, src io.Reader, h io.Writer) (int64, error) {
	var n atomic.Int64
	var lastRead atomic.Int64
	lastRead.Store(time.Now().UnixNano())
	start := time.Now()

	// The watchdog cancels the caller's context, so it has to stop when this
	// copy ends rather than when that context does.
	done := make(chan struct{})
	defer close(done)

	go func() {
		tick := time.NewTicker(time.Second)
		defer tick.Stop()
		for {
			select {
			case <-done:
				return
			case <-ctx.Done():
				return
			case <-tick.C:
				if time.Since(time.Unix(0, lastRead.Load())) > d.stallTimeout {
					cancel(errStalled)
					return
				}
				if e := time.Since(start); e > 5*time.Second {
					if m := d.speeds.median(); m > 0 && float64(n.Load())/e.Seconds() < m*0.1 {
						cancel(errSlow)
						return
					}
				}
			}
		}
	}()

	buf := make([]byte, 32*1024)
	for {
		if err := ctx.Err(); err != nil {
			if c := context.Cause(ctx); c != nil {
				return n.Load(), c
			}
			return n.Load(), err
		}

		nr, err := src.Read(buf)
		if nr > 0 {
			lastRead.Store(time.Now().UnixNano())
			nw, writeErr := dst.Write(buf[:nr])
			if nw > 0 {
				if _, hashErr := h.Write(buf[:nw]); hashErr != nil {
					return n.Load(), hashErr
				}
				d.progress.add(int64(nw))
				n.Add(int64(nw))
			}
			if writeErr != nil {
				return n.Load(), writeErr
			}
			if nw != nr {
				return n.Load(), io.ErrShortWrite
			}
		}
		if err == io.EOF {
			return n.Load(), nil
		}
		if err != nil {
			return n.Load(), err
		}
	}
}

// resolve follows redirects to find the final download URL.
// Uses GET (not HEAD) because registries may return 200 for HEAD without
// redirecting to CDN, while GET triggers the actual CDN redirect.
func (d *downloader) resolve(ctx context.Context, rawURL string) (*url.URL, error) {
	u, _ := url.Parse(rawURL)
	for range 10 {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
		req.Header.Set("User-Agent", d.userAgent)
		prev := d.authToken()
		if prev != "" {
			req.Header.Set("Authorization", "Bearer "+prev)
		}

		resp, err := d.client.Do(req)
		if err != nil {
			return nil, err
		}
		// A 200 response can be the entire multi-GB blob. Discovery only needs
		// status/headers: draining it would download the model twice.
		resp.Body.Close()

		switch resp.StatusCode {
		case http.StatusOK:
			return u, nil
		case http.StatusUnauthorized:
			if d.getToken == nil {
				return nil, fmt.Errorf("unauthorized")
			}
			ch := parseAuthChallenge(resp.Header.Get("WWW-Authenticate"))
			if err := d.refreshToken(ctx, ch, prev); err != nil {
				return nil, err
			}
		case http.StatusTemporaryRedirect, http.StatusFound, http.StatusMovedPermanently:
			loc, _ := resp.Location()
			if err := ValidateRedirectScheme(loc, d.baseURL); err != nil {
				return nil, err
			}
			if loc.Host != u.Host {
				if err := ValidateRedirectTarget(ctx, loc, d.baseURL, d.allowPrivate); err != nil {
					return nil, err
				}
				return loc, nil
			}
			u = loc
		default:
			return nil, fmt.Errorf("status %d", resp.StatusCode)
		}
	}
	return nil, fmt.Errorf("too many redirects")
}

type reservedDiskWriter struct {
	writer      io.Writer
	reservation DiskReservation
}

func (w reservedDiskWriter) Write(p []byte) (int, error) {
	if err := w.reservation.Check(); err != nil {
		return 0, err
	}
	n, err := w.writer.Write(p)
	if n > 0 {
		w.reservation.Written(int64(n))
	}
	return n, err
}

type speedTracker struct {
	mu     sync.Mutex
	speeds []float64
}

func (s *speedTracker) record(v float64) {
	s.mu.Lock()
	s.speeds = append(s.speeds, v)
	if len(s.speeds) > 30 {
		s.speeds = s.speeds[1:]
	}
	s.mu.Unlock()
}

func (s *speedTracker) median() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.speeds) < 5 {
		return 0
	}
	sorted := make([]float64, len(s.speeds))
	copy(sorted, s.speeds)
	slices.Sort(sorted)
	return sorted[len(sorted)/2]
}

const defaultStallTimeout = 10 * time.Second
