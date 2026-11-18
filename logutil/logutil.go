package logutil

import (
	"context"
	"io"
	"log/slog"
	"path/filepath"
	"runtime"
	"time"
)

const LevelTrace slog.Level = -8

func NewLogger(w io.Writer, level slog.Level) *slog.Logger {
	return slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			switch attr.Key {
			case slog.LevelKey:
				switch attr.Value.Any().(slog.Level) {
				case LevelTrace:
					attr.Value = slog.StringValue("TRACE")
				}
			case slog.SourceKey:
				if source, ok := attr.Value.Any().(*slog.Source); ok {
					source.File = filepath.Base(source.File)
				}
			}
			return attr
		},
	}))
}

// TraceEnabled lets callers skip constructing expensive diagnostic fields.
func TraceEnabled(ctx context.Context) bool {
	return slog.Default().Enabled(ctx, LevelTrace)
}

func Trace(msg string, args ...any) {
	trace(context.Background(), msg, args...)
}

func TraceContext(ctx context.Context, msg string, args ...any) {
	trace(ctx, msg, args...)
}

func trace(ctx context.Context, msg string, args ...any) {
	if logger := slog.Default(); logger.Enabled(ctx, LevelTrace) {
		var pcs [1]uintptr
		runtime.Callers(3, pcs[:]) // Skip Callers, trace and the public wrapper.
		record := slog.NewRecord(time.Now(), LevelTrace, msg, pcs[0])
		record.Add(args...)
		logger.Handler().Handle(ctx, record)
	}
}
