package imagegen

import (
	"crypto/rand"
	"encoding/hex"
	"unsafe"

	"golang.org/x/sys/windows"
)

func newWanReference(size int64) wanReference {
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return wanReference{}
	}
	name := `Local\wildllama-wan-` + hex.EncodeToString(id[:])
	ptr, _ := windows.UTF16PtrFromString(name)
	h, err := windows.CreateFileMapping(windows.InvalidHandle, nil, windows.PAGE_READWRITE, uint32(size>>32), uint32(size), ptr)
	if err != nil {
		if h != 0 {
			windows.CloseHandle(h)
		}
		return wanReference{}
	}
	return wanReference{name: name, close: func() { windows.CloseHandle(h) }, view: func(length int, use func([]byte) error) error {
		addr, err := windows.MapViewOfFile(h, windows.FILE_MAP_READ|windows.FILE_MAP_WRITE, 0, 0, uintptr(length))
		if err != nil {
			return err
		}
		defer windows.UnmapViewOfFile(addr)
		return use(unsafe.Slice((*byte)(unsafe.Pointer(addr)), length))
	}}
}
