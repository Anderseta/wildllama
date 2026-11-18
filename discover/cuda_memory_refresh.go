package discover

import (
	"context"
	"log/slog"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// Architecture validation and library selection have already completed. A live
// GGML memory sample from the same library is enough to refresh known devices.
func refreshCUDAMemory(ctx context.Context, known []ml.DeviceInfo, updated []bool, env map[string]string) bool {
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
		return false
	}
	var dirs []string
	for i, gpu := range known {
		if updated[i] {
			continue
		}
		if gpu.Library != "CUDA" || gpu.PhysicalUUID == "" || gpu.PCIID == "" || len(gpu.LibraryPath) == 0 {
			return false
		}
		if dirs == nil {
			dirs = gpu.LibraryPath
		} else if !slices.Equal(dirs, gpu.LibraryPath) {
			return false
		}
	}
	if len(dirs) == 0 {
		return false
	}
	exe, err := llm.FindLlamaServer()
	if err != nil {
		return false
	}
	// Leave time for the ordinary discovery path if this driver cannot
	// provide a usable sample.
	probeCtx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	observed, stderr, err := discoverNativeDevices(probeCtx, exe, dirs, env)
	if err != nil {
		logNativeProbeFailure(err, stderr, dirs)
		return false
	}
	if ctx.Err() != nil || !applyCUDAMemory(known, updated, observed) {
		return false
	}
	slog.Debug("refreshed CUDA memory with the native probe")
	return true
}

func applyCUDAMemory(known []ml.DeviceInfo, updated []bool, observed []nativeProbeDevice) bool {
	const mib = 1 << 20 // --list-devices reports whole MiB.
	samples := make([]uint64, len(known))
	for i, gpu := range known {
		if updated[i] {
			continue
		}
		for j, other := range known {
			if i != j && other.Library == gpu.Library && strings.EqualFold(other.PCIID, gpu.PCIID) {
				return false
			}
		}
		found := false
		for _, dev := range observed {
			if dev.Library != "CUDA" || !strings.EqualFold(dev.DeviceID, gpu.PCIID) {
				continue
			}
			// A PCI address can represent several partitions. An ambiguous or
			// driver-only sample must fall back to full device discovery.
			if found || !dev.MemoryReported || dev.PhysicalUUID == "" || !strings.EqualFold(dev.PhysicalUUID, gpu.PhysicalUUID) ||
				dev.TotalMemory/mib != gpu.TotalMemory/mib || dev.TotalMemory == 0 || dev.FreeMemory > dev.TotalMemory {
				return false
			}
			samples[i] = dev.FreeMemory / mib * mib
			found = true
		}
		if !found {
			return false
		}
	}
	for i, sample := range samples {
		if !updated[i] {
			known[i].FreeMemory = sample
			updated[i] = true
		}
	}
	return true
}
