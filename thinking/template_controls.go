package thinking

import (
	"regexp"
	"slices"
	"strings"

	"github.com/ollama/ollama/types/model"
)

var (
	jinjaComment      = regexp.MustCompile(`(?s)\{#.*?#\}`)
	jinjaRaw          = regexp.MustCompile(`(?s)\{%[-+]?\s*raw\s*[-+]?%\}.*?\{%[-+]?\s*endraw\s*[-+]?%\}`)
	jinjaStatement    = regexp.MustCompile(`(?s)\{%[-+]?\s*(.*?)\s*[-+]?%\}`)
	effortDefault     = regexp.MustCompile(`^set\s+([A-Za-z_][A-Za-z_0-9]*)\s*=\s*reasoning_effort\s*\|\s*default\s*\(\s*(?:'([A-Za-z0-9_-]+)'|"([A-Za-z0-9_-]+)")\s*\)\s*$`)
	effortOverride    = regexp.MustCompile(`^set\s+reasoning_effort\s*=`)
	jinjaAssignment   = regexp.MustCompile(`^set\s+([A-Za-z_][A-Za-z_0-9]*)\s*=`)
	thinkingDefault   = regexp.MustCompile(`^set\s+enable_thinking\s*=\s*enable_thinking\s*\|\s*default\s*\(\s*(?:true|false)\s*\)\s*$`)
	effortAllowed     = regexp.MustCompile(`^(?:if|elif)\s+([A-Za-z_][A-Za-z_0-9]*)\s+not\s+in\s*(\([^)]*\)|\[[^\]]*\])\s*$`)
	effortCondition   = regexp.MustCompile(`^(?:if|elif)\s+([A-Za-z_][A-Za-z_0-9]*)\s*==\s*(?:'([A-Za-z0-9_-]+)'|"([A-Za-z0-9_-]+)")\s*$`)
	thinkingCondition = regexp.MustCompile(`^(?:if|elif)\s+.*\benable_thinking\b`)
)

// TemplateControls discovers controls explicitly declared by a native Jinja
// template. It never executes template code and does not infer effort levels from
// model names or prompt prose. Unrecognized contracts remain unknown.
//
// Named levels require a literal default, a literal validation allowlist, and a
// branch that actually uses the resolved value. This handles newer templates
// without tying support to an architecture or a particular quantization.
func TemplateControls(source string) *model.Thinking {
	if len(source) > 1<<20 {
		return nil
	}
	source = jinjaRaw.ReplaceAllString(jinjaComment.ReplaceAllString(source, ""), "")
	if !TemplateSupportsThinking(source) {
		return nil
	}
	statements := jinjaStatement.FindAllStringSubmatch(source, -1)
	toggle := false
	for _, statement := range statements {
		code := strings.TrimSpace(statement[1])
		if match := jinjaAssignment.FindStringSubmatch(code); match != nil && match[1] == "enable_thinking" && !thinkingDefault.MatchString(code) {
			return nil // A fixed or computed override need not honor the caller.
		}
		if thinkingCondition.MatchString(statement[1]) {
			toggle = true
		}
	}
	if named := templateEffortControls(statements); named != nil {
		if toggle {
			named.Values = append([]any{false}, named.Values...)
		}
		return named
	}
	if toggle {
		// Native thinking-capable models historically receive think:true on
		// omission at the serving boundary, even if Jinja has a different default.
		return &model.Thinking{Values: []any{false, true}, Default: true}
	}
	return nil
}

func templateEffortControls(statements [][]string) *model.Thinking {
	var alias, fallback string
	for _, statement := range statements {
		code := strings.TrimSpace(statement[1])
		if match := effortDefault.FindStringSubmatch(code); match != nil {
			if alias != "" {
				return nil // Ambiguous assignments need a reviewed renderer.
			}
			alias, fallback = match[1], firstLiteral(match[2], match[3])
		} else if effortOverride.MatchString(code) {
			return nil // A hardcoded assignment would override the caller.
		}
	}
	if alias == "" {
		return nil
	}
	var levels []string
	used := false
	for _, statement := range statements {
		code := strings.TrimSpace(statement[1])
		if match := jinjaAssignment.FindStringSubmatch(code); match != nil && match[1] == alias && !effortDefault.MatchString(code) {
			return nil
		}
		if match := effortAllowed.FindStringSubmatch(code); match != nil && match[1] == alias {
			if levels != nil {
				return nil
			}
			levels = templateStringList(match[2])
			if len(levels) < 2 || !slices.Contains(levels, fallback) {
				return nil
			}
		}
		if match := effortCondition.FindStringSubmatch(code); match != nil && match[1] == alias {
			used = true
		}
	}
	if !used || len(levels) < 2 {
		return nil
	}
	controls := &model.Thinking{Default: fallback}
	for _, level := range levels {
		controls.Values = append(controls.Values, level)
	}
	return controls
}

func firstLiteral(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// Parse only simple literal names, rather than evaluating arbitrary Jinja lists.
func templateStringList(value string) []string {
	value = strings.TrimSpace(value[1 : len(value)-1])
	var result []string
	for value != "" {
		quote := value[0]
		if quote != '\'' && quote != '"' {
			return nil
		}
		end := strings.IndexByte(value[1:], quote)
		if end < 0 || end > 63 {
			return nil
		}
		name := value[1 : end+1]
		if name == "" || slices.Contains(result, name) {
			return nil
		}
		for _, ch := range name {
			if ch != '_' && ch != '-' && !(ch >= 'a' && ch <= 'z') && !(ch >= 'A' && ch <= 'Z') && !(ch >= '0' && ch <= '9') {
				return nil
			}
		}
		result = append(result, name)
		value = strings.TrimSpace(value[end+2:])
		if value == "" {
			break
		}
		if value[0] != ',' || len(result) >= 32 {
			return nil
		}
		value = strings.TrimSpace(value[1:])
	}
	return result
}
