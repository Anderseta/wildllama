//go:build windows

package imagegen

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

func conditioningFileStamp(file *os.File, info os.FileInfo) (string, bool) {
	var id windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(windows.Handle(file.Fd()), &id); err != nil {
		return "", false
	}
	var basic struct {
		Creation, Access, Write, Change int64
		Attributes, Padding             uint32
	}
	if err := windows.GetFileInformationByHandleEx(windows.Handle(file.Fd()), windows.FileBasicInfo, (*byte)(unsafe.Pointer(&basic)), uint32(unsafe.Sizeof(basic))); err != nil || basic.Change == 0 {
		return "", false
	}
	return fmt.Sprintf("%x:%x:%x:%d:%d:%d", id.VolumeSerialNumber, id.FileIndexHigh, id.FileIndexLow, basic.Change, info.Size(), info.ModTime().UnixNano()), true
}

func conditioningStampCanReuse(stamp string) bool {
	parts := strings.Split(stamp, ":")
	if len(parts) != 6 {
		return false
	}
	change, err := strconv.ParseUint(parts[3], 10, 64)
	if err != nil {
		return false
	}
	modified := windows.Filetime{LowDateTime: uint32(change), HighDateTime: uint32(change >> 32)}
	// NTFS can coalesce rapid writes into the same change-time clock tick.
	// Rehash newly changed files; installed multi-GB weights retain memo hits.
	return time.Since(time.Unix(0, modified.Nanoseconds())) >= time.Second
}
