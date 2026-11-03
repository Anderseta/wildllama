//go:build windows

package imagegen

import (
	"errors"
	"golang.org/x/sys/windows"
	"strings"
)

func isDiskFullError(err error) bool {
	return errors.Is(err, windows.ERROR_DISK_FULL) || errors.Is(err, windows.ERROR_HANDLE_DISK_FULL)
}

func diskFilesystem(path string) (string, error) {
	source, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return "", err
	}
	volume := make([]uint16, 32768)
	if err = windows.GetVolumePathName(source, &volume[0], uint32(len(volume))); err != nil {
		return "", err
	}
	return strings.ToLower(windows.UTF16ToString(volume)), nil
}

func availableDiskSpace(path string) (uint64, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	var available uint64
	err = windows.GetDiskFreeSpaceEx(p, &available, nil, nil)
	return available, err
}
