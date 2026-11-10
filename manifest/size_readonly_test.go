package manifest

import (
	"sync"
	"testing"
)

func TestManifestSizeDoesNotModifyLayerBackingArray(t *testing.T) {
	backing := []Layer{{Size: 2}, {Size: 3}, {Size: 4}, {Digest: "sentinel", Size: 99}}
	manifest := &Manifest{Layers: backing[:3], Config: Layer{Digest: "config", Size: 5}}
	var readers sync.WaitGroup
	for range 16 {
		readers.Go(func() {
			for range 100 {
				if got := manifest.Size(); got != 14 {
					t.Errorf("manifest size = %d, want 14", got)
				}
			}
		})
	}
	readers.Wait()
	if backing[3].Digest != "sentinel" || backing[3].Size != 99 {
		t.Fatal("read-only size lookup modified shared layer storage")
	}
}

func BenchmarkManifestSizeReadOnly(b *testing.B) {
	manifest := &Manifest{Layers: make([]Layer, 128), Config: Layer{Size: 100}}
	for i := range manifest.Layers {
		manifest.Layers[i].Size = 100
	}
	b.ReportAllocs()
	for b.Loop() {
		if got := manifest.Size(); got != 12900 {
			b.Fatal(got)
		}
	}
}
