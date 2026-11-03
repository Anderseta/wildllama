package imagegen

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"
)

type diskReserveKey struct{}

var ErrInsufficientDiskSpace = errors.New("insufficient free disk space")

func IsDiskSpaceError(err error) bool {
	return errors.Is(err, ErrInsufficientDiskSpace) || isDiskFullError(err)
}

func WithDiskReserve(ctx context.Context, mib int) context.Context {
	return context.WithValue(ctx, diskReserveKey{}, mib)
}

// CheckDiskSpace checks the target filesystem, including an as-yet nonexistent
// destination. It reserves space for known remaining bytes plus free headroom;
// it is a preflight, not a filesystem quota against other applications.
func CheckDiskSpace(ctx context.Context, path string, incoming int64) error {
	mib, _ := ctx.Value(diskReserveKey{}).(int)
	if mib <= 0 {
		return ctx.Err()
	}
	return checkDiskSpace(path, incoming, int64(mib)<<20, availableDiskSpace)
}

func diskDirectory(path string) (string, error) {
	ancestor := filepath.Clean(path)
	for {
		if info, err := os.Stat(ancestor); err == nil {
			if info.IsDir() {
				break
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
		parent := filepath.Dir(ancestor)
		if parent == ancestor {
			return "", errors.New("destination filesystem is unavailable")
		}
		ancestor = parent
	}
	return ancestor, nil
}

func DiskFilesystem(path string) (string, error) {
	directory, err := diskDirectory(path)
	if err != nil {
		return "", err
	}
	return diskFilesystem(directory)
}

func checkDiskSpace(path string, incoming, reserve int64, available func(string) (uint64, error)) error {
	if incoming < 0 || reserve < 0 || incoming > math.MaxInt64-reserve {
		return errors.New("invalid disk space reservation")
	}
	ancestor, err := diskDirectory(path)
	if err != nil {
		return err
	}
	free, err := available(ancestor)
	if err != nil {
		return fmt.Errorf("check image disk space: %w", err)
	}
	if free < uint64(incoming+reserve) {
		return fmt.Errorf("%w: need %d MiB for remaining data and %d MiB free headroom; available %d MiB", ErrInsufficientDiskSpace, (incoming+(1<<20)-1)>>20, reserve>>20, free>>20)
	}
	return nil
}

// monitorImageDisks stops the child before it can continue writing after free
// headroom disappears. OS allocation and concurrent programs can race polling;
// this intentionally does not claim to be a hard disk quota.
func monitorImageDisks(ctx context.Context, cancel context.CancelFunc, paths []string) <-chan error {
	result := make(chan error, 1)
	go func() {
		defer close(result)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				for _, path := range paths {
					if err := CheckDiskSpace(ctx, path, 0); err != nil && ctx.Err() == nil {
						result <- err
						cancel()
						return
					}
				}
			}
		}
	}()
	return result
}
