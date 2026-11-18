package server

import "testing"

func BenchmarkNormalize4096(b *testing.B) {
	vec := make([]float32, 4096)
	for i := range vec {
		vec[i] = float32(i%17-8) / 17
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := normalize(vec); err != nil {
			b.Fatal(err)
		}
	}
}
