//go:build windows

package imagegen

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

var getImageProcessMemoryInfo = windows.NewLazySystemDLL("kernel32.dll").NewProc("K32GetProcessMemoryInfo")

type imageProcessMemoryCounters struct {
	Size                    uint32
	PageFaultCount          uint32
	PeakWorkingSetSize      uintptr
	WorkingSetSize          uintptr
	QuotaPeakPagedPoolUsage uintptr
	QuotaPagedPoolUsage     uintptr
	QuotaPeakNonPagedUsage  uintptr
	QuotaNonPagedUsage      uintptr
	PagefileUsage           uintptr
	PeakPagefileUsage       uintptr
}

// The open process handle remains tied to the same child after it exits.
func newImageWorkingSet(pid int) (peak func() uint64, close func()) {
	if pid <= 0 || getImageProcessMemoryInfo.Find() != nil {
		return nil, nil
	}
	process, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return nil, nil
	}
	return func() uint64 {
		var counters imageProcessMemoryCounters
		counters.Size = uint32(unsafe.Sizeof(counters))
		ok, _, _ := getImageProcessMemoryInfo.Call(uintptr(process), uintptr(unsafe.Pointer(&counters)), uintptr(counters.Size))
		if ok == 0 {
			return 0
		}
		return uint64(counters.PeakWorkingSetSize)
	}, func() { _ = windows.CloseHandle(process) }
}
