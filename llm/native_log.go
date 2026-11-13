package llm

import (
	"bytes"
	"io"
	"log/slog"

	"github.com/ollama/ollama/envconfig"
)

const nativeGraphLogPrefix = "sched_graph_realloc: graph reallocation:"

// nativeLogFilter removes a known repetitive diagnostic before downstream
// parsing and disk output. It never buffers lines: an incomplete prefix is
// forwarded intact, so even a process exiting mid-line cannot hide an error.
// stdout/stderr share this writer and os/exec serializes Write calls.
type nativeLogFilter struct {
	inner        io.Writer
	continuation bool
	discard      bool
}

func newNativeLogFilter(out io.Writer) io.Writer {
	if envconfig.LogLevel() <= slog.LevelDebug {
		return out
	}
	return &nativeLogFilter{inner: out}
}

func (w *nativeLogFilter) Write(b []byte) (int, error) {
	total := len(b)
	for len(b) > 0 {
		n := len(b)
		complete := false
		if i := bytes.IndexByte(b, '\n'); i >= 0 {
			n, complete = i+1, true
		}
		line := b[:n]
		if !w.continuation {
			w.discard = bytes.HasPrefix(line, []byte(nativeGraphLogPrefix))
		}
		if !w.discard {
			written, err := w.inner.Write(line)
			if err == nil && written != len(line) {
				err = io.ErrShortWrite
			}
			if err != nil {
				return total - len(b) + written, err
			}
		}
		b = b[n:]
		w.continuation = !complete
		if complete {
			w.discard = false
		}
	}
	return total, nil
}
