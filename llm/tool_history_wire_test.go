package llm

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func BenchmarkNativeToolHistoryWire(b *testing.B) {
	runner := &llamaServerRunner{}
	args := api.NewToolCallFunctionArguments()
	args.Set("path", "model.go")
	args.Set("content", strings.Repeat("if a < b && c > 0 {\n    save(\"ação\")\n}\n", 160))
	args.Set("seed", int64(5232021495785965240))
	call := api.ToolCall{ID: "tool-1", Function: api.ToolCallFunction{Name: "write", Arguments: args}}
	props := api.NewToolPropertiesMap()
	props.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	props.Set("content", api.ToolProperty{Type: api.PropertyType{"string"}})
	tool := api.Tool{Type: "function", Function: api.ToolFunction{Name: "write", Parameters: api.ToolFunctionParameters{Type: "object", Properties: props}}}
	history := ChatRequest{Tools: []api.Tool{tool}}
	for range 24 {
		history.Messages = append(history.Messages, api.Message{Role: "assistant", ToolCalls: []api.ToolCall{call}}, api.Message{Role: "tool", ToolCallID: "tool-1", Content: "Saved"})
	}
	b.ReportAllocs()
	for b.Loop() {
		req, err := runner.llamaServerChatRequest(history, true)
		if err != nil {
			b.Fatal(err)
		}
		wire, err := encodeLlamaServerChatRequest(req)
		if err != nil || wire.Len() == 0 {
			b.Fatalf("encoding: %v", err)
		}
	}
}

func TestNativeToolHistoryWireEscapesAndOrder(t *testing.T) {
	args := api.NewToolCallFunctionArguments()
	args.Set("z\"<&\u2028", "<a href=\"x\">ação & 🙂</a>\n")
	args.Set("seed", int64(5232021495785965240))
	calls, err := llamaServerChatToolCalls([]api.ToolCall{{Function: api.ToolCallFunction{Name: "test", Arguments: args}}})
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"z\"\u003c\u0026\u2028":"\u003ca href=\"x\"\u003eação \u0026 🙂\u003c/a\u003e\n","seed":5232021495785965240}`
	if got := calls[0].Function.Arguments; got != want {
		t.Fatalf("changed arguments: %s", got)
	}
	args.Set("unsupported", make(chan int))
	if _, err := json.Marshal(args); err == nil {
		t.Fatal("accepted unencodable arguments")
	}
}

func TestCompletionKeepsCallerOutputBudget(t *testing.T) {
	runner, _, _ := terminalTestRunner(t, "data: {\"stop\":true,\"stop_type\":\"eos\"}\n\n", nil)
	options := api.DefaultOptions()
	options.NumPredict = -1
	if err := runner.Completion(t.Context(), CompletionRequest{Options: &options}, func(CompletionResponse) {}); err != nil {
		t.Fatal(err)
	}
	if options.NumPredict != -1 {
		t.Fatalf("caller budget changed to %d", options.NumPredict)
	}
}
