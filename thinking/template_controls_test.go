package thinking

import (
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestTemplateControlsPublishedQwenTemplates(t *testing.T) {
	for _, tc := range []struct {
		file string
		want *model.Thinking
	}{
		{"../model/renderers/testdata/qwen38_chat_template.jinja", &model.Thinking{Values: []any{false, "xhigh", "medium", "low"}, Default: "xhigh"}},
		{"../server/testdata/qwen35_chat_template.jinja", &model.Thinking{Values: []any{false, true}, Default: true}},
	} {
		t.Run(tc.file, func(t *testing.T) {
			source, err := os.ReadFile(tc.file)
			if err != nil {
				t.Fatal(err)
			}
			if got := TemplateControls(string(source)); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("controls = %#v, want %#v", got, tc.want)
			}
		})
	}
}

func TestTemplateControlsRequireDeclaredContract(t *testing.T) {
	const contract = `{% set resolved = reasoning_effort | default("deep") %}
{% if resolved not in ["quick", "deep"] %}{{ raise_exception('unsupported') }}{% endif %}
{% if resolved == "deep" %}think carefully{% endif %}<think></think>`
	for _, tc := range []struct {
		name, source string
		want         *model.Thinking
	}{
		{"future names", contract, &model.Thinking{Values: []any{"quick", "deep"}, Default: "deep"}},
		{"future names with toggle", contract + `{% if enable_thinking %}yes{% endif %}`, &model.Thinking{Values: []any{false, "quick", "deep"}, Default: "deep"}},
		{"prompt prose", `reasoning_effort defaults deep and supports quick/deep <think></think>`, nil},
		{"commented contract", "{# " + contract + " #}", nil},
		{"raw example", "{% raw %}" + contract + "{% endraw %}", nil},
		{"no thinking output", strings.ReplaceAll(strings.ReplaceAll(contract, "<think>", ""), "</think>", ""), nil},
		{"dynamic allowlist", strings.ReplaceAll(contract, `["quick", "deep"]`, "allowed_efforts"), nil},
		{"duplicate allowlist", strings.ReplaceAll(contract, `["quick", "deep"]`, `["deep", "deep"]`), nil},
		{"unknown default", strings.ReplaceAll(contract, `default("deep")`, `default("unknown")`), nil},
		{"unused declaration", strings.ReplaceAll(contract, `{% if resolved == "deep" %}think carefully{% endif %}`, ""), nil},
		{"hardcoded effort", `{% set reasoning_effort = 'deep' %}` + contract, nil},
		{"hardcoded thinking on", `{% set enable_thinking = true %}{% if enable_thinking %}` + contract + `{% endif %}`, nil},
		{"hardcoded thinking off", `{% set enable_thinking = false %}{% if enable_thinking %}` + contract + `{% endif %}`, nil},
		{"thinking overrides false", `{% set enable_thinking = enable_thinking|default(true, true) %}{% if enable_thinking %}` + contract + `{% endif %}`, nil},
		{"thinking default respects caller", `{% set enable_thinking = enable_thinking|default(true) %}{% if enable_thinking %}` + contract + `{% endif %}`, &model.Thinking{Values: []any{false, "quick", "deep"}, Default: "deep"}},
		{"overridden alias", contract + `{% set resolved = 'quick' %}`, nil},
		{"overridden earlier alias", `{% set resolved = 'quick' %}` + contract, nil},
		{"ambiguous defaults", contract + `{% set other = reasoning_effort|default('quick') %}`, nil},
		{"oversized template", contract + strings.Repeat(" ", 1<<20), nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := TemplateControls(tc.source); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("controls = %#v, want %#v", got, tc.want)
			}
		})
	}
}
