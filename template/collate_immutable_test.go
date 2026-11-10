package template

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestCollatePreservesHistoryAndMetadata(t *testing.T) {
	messages := []api.Message{
		{Role: "system", Content: "um"}, {Role: "system", Content: "dois"},
		{Role: "user", Content: "ação", Thinking: "metadata", Images: []api.ImageData{[]byte("image")}},
		{Role: "user", Content: ""}, {Role: "user", Content: "fim"},
		{Role: "tool", Content: "first", ToolCallID: "one"}, {Role: "tool", Content: "second", ToolCallID: "two"},
	}
	before, _ := json.Marshal(messages)
	for range 3 {
		system, result := collate(messages)
		if system != "um\n\ndois" || len(result) != 4 || result[0].Content != system || result[1].Content != "ação\n\n\n\nfim" || result[1].Thinking != "metadata" || len(result[1].Images) != 1 || result[2].ToolCallID != "one" || result[3].ToolCallID != "two" {
			t.Fatalf("merged message content or metadata changed: system=%q messages=%+v", system, result)
		}
		after, _ := json.Marshal(messages)
		if !bytes.Equal(before, after) {
			t.Fatal("render preparation modified the original conversation")
		}
	}
}

func TestExecuteSharedHistoryIsStable(t *testing.T) {
	tmpl, err := Parse(`{{range .Messages}}{{.Role}}:{{.Content}};{{end}}`)
	if err != nil {
		t.Fatal(err)
	}
	messages := []api.Message{{Role: "user", Content: "um"}, {Role: "user", Content: "dois"}, {Role: "assistant", Content: "ok"}}
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 3 {
				var output strings.Builder
				if err := tmpl.Execute(&output, Values{Messages: messages}); err != nil {
					t.Error(err)
				} else if output.String() != "user:um\n\ndois;assistant:ok;" {
					t.Errorf("repeated rendering duplicated or changed history: %q", output.String())
				}
			}
		})
	}
	wg.Wait()
}
