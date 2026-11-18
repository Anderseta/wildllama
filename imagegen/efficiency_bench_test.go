package imagegen

import (
	"os"
	"testing"
)

// Set WILDLLAMA_BENCH_ENCODER to measure repeated validation of an installed encoder.
func BenchmarkInspectInstalledEncoder(b *testing.B) {
	path := os.Getenv("WILDLLAMA_BENCH_ENCODER")
	if path == "" {
		b.Skip("supply a local encoder")
	}
	first, err := InspectLocalComponent(path, "text_encoder")
	if err != nil || first.Compatibility != "supported" || first.TensorCount == 0 {
		b.Fatal(first, err)
	}
	b.ReportAllocs()
	for b.Loop() {
		got, err := InspectLocalComponent(path, "text_encoder")
		if err != nil || got.TensorCount != first.TensorCount || got.SupportsEditing != first.SupportsEditing {
			b.Fatal(got, err)
		}
	}
}

func BenchmarkInstalledImageRunnerStatus(b *testing.B) {
	path := os.Getenv("WILDLLAMA_BENCH_IMAGE_RUNNER")
	if path == "" {
		b.Skip("supply a pinned image runner")
	}
	b.ReportAllocs()
	for b.Loop() {
		status := (&Engine{config: Config{RunnerPath: path}}).Status()
		if !status.Ready {
			b.Fatal(status.Error)
		}
	}
}
