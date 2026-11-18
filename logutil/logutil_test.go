package logutil

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"runtime"
	"strings"
	"testing"
)

func TestTraceReportsExternalCaller(t *testing.T) {
	original := slog.Default()
	t.Cleanup(func() { slog.SetDefault(original) })
	var out bytes.Buffer
	slog.SetDefault(NewLogger(&out, LevelTrace))

	_, _, line, _ := runtime.Caller(0)
	Trace("plain")
	TraceContext(context.Background(), "with context")
	for _, want := range []int{line + 1, line + 2} {
		if !strings.Contains(out.String(), fmt.Sprintf("logutil_test.go:%d", want)) {
			t.Fatalf("missing caller line %d in %s", want, &out)
		}
	}
}

func TestNewLoggerAllowsSourceAttr(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelDebug)

	logger.Debug("message", "source", "runtime")

	if !strings.Contains(buf.String(), `source=runtime`) {
		t.Fatalf("expected user source attr in log, got %q", buf.String())
	}
}
