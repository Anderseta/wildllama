//go:build !windows

package imagegen

import (
	"fmt"
	"os"
	"reflect"
)

func conditioningStampCanReuse(_ string) bool { return true }

func conditioningFileStamp(_ *os.File, info os.FileInfo) (string, bool) {
	value := reflect.Indirect(reflect.ValueOf(info.Sys()))
	if !value.IsValid() || value.Kind() != reflect.Struct {
		return "", false
	}
	device, inode := value.FieldByName("Dev"), value.FieldByName("Ino")
	change := value.FieldByName("Ctim")
	if !change.IsValid() {
		change = value.FieldByName("Ctimespec")
	}
	if !device.IsValid() || !inode.IsValid() || !change.IsValid() || !device.CanInterface() || !inode.CanInterface() || !change.CanInterface() {
		// On filesystems without a reliable change stamp, recompute the content
		// digest rather than risk reusing stale conditioning.
		return "", false
	}
	return fmt.Sprintf("%v:%v:%v:%d:%d", device.Interface(), inode.Interface(), change.Interface(), info.Size(), info.ModTime().UnixNano()), true
}
