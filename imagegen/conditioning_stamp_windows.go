//go:build windows

package imagegen

import (
	"fmt"
	"os"
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
