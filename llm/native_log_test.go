package llm

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
)

func TestNativeLogFilterChunks(t *testing.T) {
	graph := nativeGraphLogPrefix + " nodes = 70 (changed = 0), splits = 1\n"
	before := "load_tensors: offloaded 66/66 layers to GPU\n"
	after := "llama_context: CUDA0 compute buffer size = 64.00 MiB\n" +
		"wildllama_activity {\"kind\":\"phase\",\"phase\":\"ready\"}\n" +
		"CUDA error: out of memory\npartial error without newline"
	input := before + graph + after
	for split := 0; split <= len(input); split++ {
		var output bytes.Buffer
		w := &nativeLogFilter{inner: &output}
		for _, part := range []string{input[:split], input[split:]} {
			if n, err := w.Write([]byte(part)); err != nil || n != len(part) {
				t.Fatalf("split %d: Write = %d, %v", split, n, err)
			}
		}
		// Only a prefix split across pipe writes is deliberately passed through.
		want := before + after
		if split > len(before) && split < len(before)+len(nativeGraphLogPrefix) {
			want = input
		}
		if output.String() != want {
			t.Fatalf("split %d: output %q, want %q", split, output.String(), want)
		}
	}
}

func TestNativeLogFilterPreservesUnterminatedAndLongLines(t *testing.T) {
	for _, line := range []string{"s", "sched_graph_realloc:", "error: " + strings.Repeat("x", 128*1024)} {
		var output bytes.Buffer
		w := &nativeLogFilter{inner: &output}
		for _, b := range []byte(line) {
			if _, err := w.Write([]byte{b}); err != nil {
				t.Fatal(err)
			}
		}
		if output.String() != line {
			t.Fatal("partial line changed")
		}
	}
}

func TestNativeLogFilterDebug(t *testing.T) {
	t.Setenv("OLLAMA_DEBUG", "1")
	var output bytes.Buffer
	if newNativeLogFilter(&output) != &output {
		t.Fatal("debug logs must bypass filtering")
	}
}

type nativeLogFailWriter struct{ err error }

func (w nativeLogFailWriter) Write([]byte) (int, error) { return 0, w.err }

func TestNativeLogFilterWriteError(t *testing.T) {
	failed := errors.New("disk write failed")
	for _, failure := range []error{failed, nil} {
		w := &nativeLogFilter{inner: nativeLogFailWriter{failure}}
		want := failure
		if want == nil {
			want = io.ErrShortWrite
		}
		if n, err := w.Write([]byte("CUDA error: failure\n")); n != 0 || !errors.Is(err, want) {
			t.Fatalf("Write = %d, %v, want 0, %v", n, err, want)
		}
	}
}

func BenchmarkNativeLogFilter(b *testing.B) {
	line := []byte(nativeGraphLogPrefix + " nodes = 70 (changed = 0), splits = 1\n")
	for _, filtered := range []bool{false, true} {
		name := "unfiltered"
		if filtered {
			name = "filtered"
		}
		b.Run(name, func(b *testing.B) {
			var w io.Writer = &memoryParsingWriter{inner: NewStatusWriter(io.Discard), runner: &llamaServerRunner{}}
			if filtered {
				w = &nativeLogFilter{inner: w}
			}
			b.ReportAllocs()
			for b.Loop() {
				_, _ = w.Write(line)
			}
		})
	}
}
