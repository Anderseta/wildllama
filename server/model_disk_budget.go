package server

import (
	"context"
	"errors"
	"io"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/imagegen"
	"github.com/ollama/ollama/transfer"
)

type modelDiskBudget struct {
	mu         sync.Mutex
	active     map[*modelDiskReservation]bool
	check      func(context.Context, string, int64) error
	filesystem func(string) (string, error)
}

type modelDiskReservation struct {
	owner     *modelDiskBudget
	ctx       context.Context
	path, key string
	remaining atomic.Int64
	closed    bool
	lastCheck time.Time
}

var modelTransfersDisk = modelDiskBudget{active: map[*modelDiskReservation]bool{}, check: imagegen.CheckDiskSpace, filesystem: imagegen.DiskFilesystem}

func (b *modelDiskBudget) reserve(ctx context.Context, path string, size int64) (*modelDiskReservation, error) {
	mib := envconfig.MinFreeDiskMiB()
	if mib == 0 {
		return nil, ctx.Err()
	}
	if mib > 1024*1024 {
		mib = 1024
	}
	ctx = imagegen.WithDiskReserve(ctx, int(mib))
	key, err := b.filesystem(path)
	if err != nil {
		return nil, err
	}
	reservation := &modelDiskReservation{owner: b, ctx: ctx, path: path, key: key}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.active == nil {
		b.active = map[*modelDiskReservation]bool{}
	}
	if err = reservation.checkLocked(size); err != nil {
		return nil, err
	}
	reservation.remaining.Store(size)
	b.active[reservation] = true
	return reservation, nil
}

type tensorDiskReservation struct{ value *modelDiskReservation }

func (r tensorDiskReservation) Check() error    { return r.value.check() }
func (r tensorDiskReservation) Written(n int64) { r.value.consume(n) }
func (r tensorDiskReservation) Close()          { r.value.close() }
func reserveTensorDisk(ctx context.Context, path string, size int64) (transfer.DiskReservation, error) {
	value, err := modelTransfersDisk.reserve(ctx, path, size)
	if err != nil {
		return nil, err
	}
	return tensorDiskReservation{value}, nil
}

func (r *modelDiskReservation) checkLocked(additional int64) error {
	if err := r.ctx.Err(); err != nil {
		return err
	}
	if additional < 0 {
		return errors.New("invalid model disk reservation")
	}
	total := additional
	for other := range r.owner.active {
		if other.key == r.key {
			n := other.remaining.Load()
			if n > math.MaxInt64-total {
				return errors.New("model disk reservation overflow")
			}
			total += n
		}
	}
	return r.owner.check(r.ctx, r.path, total)
}

func (r *modelDiskReservation) grow(n int64) error {
	if r == nil {
		return nil
	}
	r.owner.mu.Lock()
	defer r.owner.mu.Unlock()
	if err := r.checkLocked(n); err != nil {
		return err
	}
	r.remaining.Add(n)
	return nil
}

func (r *modelDiskReservation) check() error {
	if r == nil {
		return nil
	}
	r.owner.mu.Lock()
	defer r.owner.mu.Unlock()
	if time.Since(r.lastCheck) < time.Second {
		return r.ctx.Err()
	}
	if err := r.checkLocked(0); err != nil {
		return err
	}
	r.lastCheck = time.Now()
	return nil
}

func (r *modelDiskReservation) consume(n int64) {
	if r == nil {
		return
	}
	for {
		current := r.remaining.Load()
		next := max(0, current-n)
		if r.remaining.CompareAndSwap(current, next) {
			return
		}
	}
}

func (r *modelDiskReservation) close() {
	if r == nil {
		return
	}
	r.owner.mu.Lock()
	defer r.owner.mu.Unlock()
	if !r.closed {
		delete(r.owner.active, r)
		r.closed = true
	}
}

type modelDiskWriter struct {
	writer      io.Writer
	reservation *modelDiskReservation
	part        *blobDownloadPart
	position    int64
}

func (w *modelDiskWriter) Write(p []byte) (int, error) {
	if err := w.reservation.check(); err != nil {
		return 0, err
	}
	n, err := w.writer.Write(p)
	w.position += int64(n)
	if w.position > w.part.diskWritten {
		w.reservation.consume(w.position - w.part.diskWritten)
		w.part.diskWritten = w.position
	}
	return n, err
}

type modelDiskReader struct {
	source      io.Reader
	reservation *modelDiskReservation
	unknown     bool
	pending     int64
}

func (r *modelDiskReader) Read(p []byte) (int, error) {
	r.reservation.consume(r.pending)
	r.pending = 0
	if err := r.reservation.check(); err != nil {
		return 0, err
	}
	n, err := r.source.Read(p)
	if n > 0 {
		if r.unknown {
			if spaceErr := r.reservation.grow(int64(n)); spaceErr != nil {
				return 0, spaceErr
			}
		}
		r.pending = int64(n)
	}
	return n, err
}
