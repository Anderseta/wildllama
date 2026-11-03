//go:build !windows

package imagegen

import (
	"errors"
	"fmt"
	"golang.org/x/sys/unix"
)

func isDiskFullError(err error) bool {
	return errors.Is(err, unix.ENOSPC) || errors.Is(err, unix.EDQUOT)
}

func diskFilesystem(path string) (string, error) {
	var stat unix.Stat_t
	if err := unix.Stat(path, &stat); err != nil {
		return "", err
	}
	return fmt.Sprintf("device:%d", stat.Dev), nil
}

func availableDiskSpace(path string) (uint64, error) {
	var stat unix.Statfs_t
	if err := unix.Statfs(path, &stat); err != nil {
		return 0, err
	}
	return uint64(stat.Bavail) * uint64(stat.Bsize), nil
}
