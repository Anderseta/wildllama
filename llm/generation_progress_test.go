package llm

import (
	"testing"
	"time"
)

func TestGenerationProgressBoundedAndRequestLocal(t *testing.T) {
	var first, second GenerationProgressEmitter
	if p := first.Next("tool_call"); p == nil || p.Phase != "tool_call" {
		t.Fatalf("first fragment must be visible: %+v", p)
	}
	for range 10000 {
		if p := first.Next("tool_call"); p != nil {
			t.Fatal("token-sized deltas flooded the public stream")
		}
	}
	if second.Next("tool_call") == nil {
		t.Fatal("another request inherited the rate limit")
	}
	first.last = time.Now().Add(-generationProgressInterval - time.Millisecond)
	if p := first.Next("buffered_output"); p == nil || p.Phase != "buffered_output" {
		t.Fatalf("later real output did not publish progress: %+v", p)
	}
}

func BenchmarkGenerationProgressThrottled(b *testing.B) {
	emitter := GenerationProgressEmitter{last: time.Now()}
	b.ReportAllocs()
	for b.Loop() {
		emitter.Next("tool_call")
	}
}
