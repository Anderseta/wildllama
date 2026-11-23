package imagegen

import (
	"crypto/rand"
	"encoding/hex"

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
	return wanReference{name: name, close: func() { windows.CloseHandle(h) }}
}
