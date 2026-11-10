package server

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

type promptRenderProbe struct{ calls *int }

func (p promptRenderProbe) MarshalJSON() ([]byte, error) {
	*p.calls++
	return []byte(`"tracked"`), nil
}

func TestChatPromptReusesExactlyCountedText(t *testing.T) {
	tmpl, err := template.Parse(`{{range .Messages}}{{.Role}}:{{.Content}}{{json .ToolCalls}};{{end}}`)
	if err != nil {
		t.Fatal(err)
	}
	for _, truncate := range []bool{true, false} {
		calls := 0
		args := api.NewToolCallFunctionArguments()
		args.Set("probe", promptRenderProbe{calls: &calls})
		messages := []api.Message{
			{Role: "assistant", ToolCalls: []api.ToolCall{{Function: api.ToolCallFunction{Name: "lookup", Arguments: args}}}},
			{Role: "user", Content: "continue"},
		}
		var counted string
		count := func(_ context.Context, text string) ([]int, error) { counted = text; return []int{1}, nil }
		prompt, media, err := chatPrompt(t.Context(), &Model{Template: tmpl}, count, &api.Options{Runner: api.Runner{NumCtx: 4096}}, messages, nil, nil, truncate)
		if err != nil || len(media) != 0 || calls != 1 {
			t.Fatalf("truncate=%v renders=%d media=%d error=%v", truncate, calls, len(media), err)
		}
		if truncate && prompt != counted {
			t.Fatal("the submitted prompt differs from the text whose tokens were counted")
		}
	}
}

func BenchmarkChatPromptPreparedText(b *testing.B) {
	benchmarkChatPromptText(b, false)
}

func BenchmarkChatPromptAlternatingText(b *testing.B) {
	benchmarkChatPromptText(b, true)
}

func benchmarkChatPromptText(b *testing.B, alternating bool) {
	tmpl, err := template.Parse(`{{json .Tools}}{{range .Messages}}{{.Role}}:{{.Content}};{{end}}`)
	if err != nil {
		b.Fatal(err)
	}
	m := &Model{Template: tmpl}
	messages := make([]api.Message, 128)
	for i := range messages {
		messages[i] = api.Message{Role: "user", Content: strings.Repeat("texto código ação contexto; ", 32)}
		if alternating && i%2 != 0 {
			messages[i].Role = "assistant"
		}
	}
	opts := &api.Options{Runner: api.Runner{NumCtx: 65536}}
	count := func(context.Context, string) ([]int, error) { return []int{1}, nil }
	b.ReportAllocs()
	for b.Loop() {
		prompt, _, err := chatPrompt(b.Context(), m, count, opts, messages, nil, nil, true)
		if err != nil || prompt == "" {
			b.Fatalf("prompt preparation failed: %v", err)
		}
	}
}
