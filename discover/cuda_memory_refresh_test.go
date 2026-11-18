package discover

import (
	"github.com/ollama/ollama/ml"
	"slices"
	"strings"
	"testing"
)

func TestCUDAMemoryRefreshRequiresFreshUnambiguousSamples(t *testing.T) {
	const mib = 1 << 20
	a := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "CUDA", ID: "0"}, PCIID: "0000:01:00.0", PhysicalUUID: "GPU-A", TotalMemory: 16 * mib, FreeMemory: 5 * mib}
	b := a
	b.ID = "1"
	b.PCIID = "0000:02:00.0"
	b.PhysicalUUID = "GPU-B"
	sample := func(g ml.DeviceInfo, free uint64) nativeProbeDevice {
		return nativeProbeDevice{Library: "CUDA", DeviceID: g.PCIID, PhysicalUUID: strings.ToLower(g.PhysicalUUID), MemoryReported: true, TotalMemory: g.TotalMemory + 123, FreeMemory: free}
	}
	for _, tc := range []struct {
		name   string
		mutate func([]nativeProbeDevice)
		valid  bool
	}{
		{"matching", func([]nativeProbeDevice) {}, true},
		{"unavailable", func(v []nativeProbeDevice) { v[0].MemoryReported = false }, false},
		{"wrong UUID", func(v []nativeProbeDevice) { v[0].PhysicalUUID = "another" }, false},
		{"ambiguous bus", func(v []nativeProbeDevice) { v[0].DeviceID = v[1].DeviceID }, false},
		{"wrong total", func(v []nativeProbeDevice) { v[0].TotalMemory -= mib }, false},
		{"impossible free", func(v []nativeProbeDevice) { v[0].FreeMemory = v[0].TotalMemory + 1 }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := []ml.DeviceInfo{a, b}
			updated := []bool{false, false}
			// Enumeration order can change and zero is a valid free-memory reading.
			obs := []nativeProbeDevice{sample(b, 0), sample(a, 8*mib+999)}
			tc.mutate(obs)
			if got := applyCUDAMemory(g, updated, obs); got != tc.valid {
				t.Fatal("refresh", got)
			}
			if tc.valid {
				if g[0].FreeMemory != 8*mib || g[1].FreeMemory != 0 || !slices.Equal(updated, []bool{true, true}) {
					t.Fatal("bad memory", g, updated)
				}
			} else if !slices.Equal(updated, []bool{false, false}) || g[0].FreeMemory != a.FreeMemory || g[1].FreeMemory != b.FreeMemory {
				t.Fatal("partial observation published")
			}
		})
	}
	known := []ml.DeviceInfo{a, b}
	updated := []bool{true, false}
	if !applyCUDAMemory(known, updated, []nativeProbeDevice{sample(b, 0)}) || known[0].FreeMemory != a.FreeMemory {
		t.Fatal("active runner observation replaced")
	}
	b.PCIID = a.PCIID
	if sameRefreshDevice(a, b) || applyCUDAMemory([]ml.DeviceInfo{a, b}, []bool{false, false}, []nativeProbeDevice{sample(a, 0)}) {
		t.Fatal("two partitions on one bus were treated as one GPU")
	}
}
