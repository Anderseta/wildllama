package anthropic

import (
	"bytes"
	"log/slog"

	"github.com/ollama/ollama/logutil"
	"strings"
	"testing"
)

func TestMessagesTraceFollowsLogLevel(t *testing.T) {
	original := slog.Default()
	t.Cleanup(func() { slog.SetDefault(original) })
	var out bytes.Buffer
	level := &slog.LevelVar{}
	slog.SetDefault(slog.New(slog.NewTextHandler(&out, &slog.HandlerOptions{Level: level})))
	text := "trace fixture"
	req := MessagesRequest{Model: "test", Messages: []MessageParam{{Role: "user", Content: []ContentBlock{{Type: "text", Text: &text}}}}}
	for _, enabled := range []bool{false, true} {
		if enabled {
			level.Set(logutil.LevelTrace)
		}
		out.Reset()
		got, err := FromMessagesRequest(req)
		if err != nil || len(got.Messages) != 1 || got.Messages[0].Content != text {
			t.Fatalf("converted request: %v, %v", got, err)
		}
		for _, field := range []string{"anthropic: converting request", "anthropic: converted block message", "anthropic: converted request", text} {
			if strings.Contains(out.String(), field) != enabled {
				t.Fatalf("trace enabled=%v, field=%q, log=%s", enabled, field, &out)
			}
		}
	}
}

func BenchmarkMessagesHistoryConversion(b *testing.B) {
	body := strings.Repeat("texto com acentuação e código\n", 48)
	req := MessagesRequest{Model: "test", Messages: make([]MessageParam, 64)}
	for i := range req.Messages {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		req.Messages[i] = MessageParam{Role: role, Content: []ContentBlock{{Type: "text", Text: &body}}}
	}
	b.ReportAllocs()
	for b.Loop() {
		got, err := FromMessagesRequest(req)
		if err != nil || len(got.Messages) != len(req.Messages) {
			b.Fatal(err)
		}
	}
}
