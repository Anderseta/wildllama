package imagegen

import (
	"os"

	"golang.org/x/sys/unix"
)

func newWanReference(size int64) wanReference {
	fd, err := unix.MemfdCreate("wildllama-wan", unix.MFD_CLOEXEC)
	if err != nil {
		return wanReference{}
	}
	file := os.NewFile(uintptr(fd), "wan-reference")
	if err := file.Truncate(size); err != nil {
		file.Close()
		return wanReference{}
	}
	// os/exec installs the first ExtraFiles entry at descriptor 3 in the child.
	return wanReference{name: "fd:3", file: file, close: func() { file.Close() }, view: func(length int, use func([]byte) error) error {
		data, err := unix.Mmap(fd, 0, length, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
		if err != nil {
			return err
		}
		defer unix.Munmap(data)
		return use(data)
	}}
}
