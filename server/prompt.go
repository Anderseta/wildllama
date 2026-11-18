package server

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/template"
)

type tokenizeFunc func(context.Context, string) ([]int, error)

// chatPrompt accepts a list of messages and returns the prompt and media that should be used for the next chat turn.
// chatPrompt truncates any messages that exceed the context window of the model, making sure to always include 1) the
// latest message and 2) system and developer instructions
func chatPrompt(ctx context.Context, m *Model, tokenize tokenizeFunc, opts *api.Options, msgs []api.Message, tools []api.Tool, think *api.ThinkValue, truncate bool) (prompt string, media []llm.MediaData, _ error) {
	currMsgIdx := 0
	countedStart := -1
	var countedPrompt string

	if truncate {
		textOnly := !slices.ContainsFunc(msgs, func(message api.Message) bool { return len(message.Images) > 0 })
		var budget chatMediaBudget
		var err error
		if !textOnly {
			budget, err = estimateChatMedia(ctx, m, msgs, true)
			if err != nil {
				return "", nil, err
			}
		}
		currMsgIdx, err = fittingPromptStart(ctx, msgs, opts.NumCtx, func(valid []bool) int {
			return monotonicPromptLimit(m, msgs, false, valid)
		}, func(i int) (int, error) {
			// Retain only the latest candidate, never every truncation attempt.
			countedPrompt, countedStart = "", -1
			candidate := msgs
			// Go templates read the original history. Isolate custom renderers.
			if i > 0 || m.Config.Renderer != "" {
				candidate = promptMessagesFrom(msgs, i)
			}
			p, err := renderPrompt(m, candidate, tools, think)
			if err != nil {
				return 0, err
			}
			if textOnly {
				countedPrompt, countedStart = p, i
			}
			s, err := tokenize(ctx, p)
			if err != nil {
				return 0, err
			}
			mediaTokens := budget.at(i)
			if mediaTokens > math.MaxInt-len(s) {
				return math.MaxInt, nil
			}
			return len(s) + mediaTokens, nil
		})
		if err != nil {
			return "", nil, err
		}
	}

	if currMsgIdx > 0 {
		slog.Debug("truncating input messages which exceed context length", "start", currMsgIdx)
	}
	if countedStart == currMsgIdx {
		// Text-only preparation need not tag media or render the same history
		// a second time. Submit exactly the prompt whose tokens were counted.
		return countedPrompt, nil, nil
	}
	countedPrompt = ""

	selected := msgs
	if currMsgIdx > 0 {
		selected = promptMessagesFrom(msgs, currMsgIdx)
	}
	renderMsgs, media, err := imageTaggedMessages(m, selected, false)
	if err != nil {
		return "", nil, err
	}

	p, err := renderPrompt(m, renderMsgs, tools, think)
	if err != nil {
		return "", nil, err
	}

	return p, media, nil
}

// A tool response must retain the assistant call that produced it.
func promptTruncationStarts(messages []api.Message) []bool {
	changes := make([]int, len(messages)+1)
	owners := make(map[string]int)
	lastOwner := -1
	for index, message := range messages {
		if message.Role == "user" {
			lastOwner = -1
		}
		if message.Role == "assistant" && len(message.ToolCalls) > 0 {
			lastOwner = index
			for _, call := range message.ToolCalls {
				if call.ID != "" {
					owners[call.ID] = index
				}
			}
		}
		if message.Role == "tool" {
			owner := lastOwner
			if message.ToolCallID != "" {
				if found, exists := owners[message.ToolCallID]; exists {
					owner = found
				}
			}
			if owner >= 0 && owner < index {
				changes[owner+1]++
				changes[index+1]--
			}
		}
	}
	result := make([]bool, len(messages))
	active := 0
	for index := range messages {
		active += changes[index]
		result[index] = active == 0
	}
	return result
}

func imageTaggedMessages(m *Model, msgs []api.Message, clearImages bool) ([]api.Message, []llm.MediaData, error) {
	renderMsgs := slices.Clone(msgs)
	var media []llm.MediaData

	for cnt, msg := range renderMsgs {
		if slices.Contains(m.Config.ModelFamilies, "mllama") && len(msg.Images) > 1 {
			return nil, nil, errors.New("this model only supports one image while more than one image requested")
		}

		firstID := len(media)
		for _, data := range msg.Images {
			media = append(media, llm.NewMediaData(len(media), data))
		}
		if m.Config.Renderer == "" && len(msg.Images) > 0 {
			renderMsgs[cnt].Content = tagMediaPrompt(msg.Content, firstID, len(msg.Images))
		}
		if clearImages {
			renderMsgs[cnt].Images = nil
		}
	}

	return renderMsgs, media, nil
}

// Extra media go before the original prompt; numbered markers replace existing
// [img] slots from left to right. Both passes are linear in the original text.
func tagMediaPrompt(prompt string, firstID, count int) string {
	const marker = "[img]"
	rest := prompt
	placed := 0
	for placed < count {
		_, tail, found := strings.Cut(rest, marker)
		if !found {
			break
		}
		placed++
		rest = tail
	}
	var out strings.Builder
	tagSize := len("[img-]") + len(strconv.Itoa(firstID+count-1))
	out.Grow(len(prompt) - len(marker)*placed + tagSize*count)
	writeTag := func(id int) {
		var digits [20]byte
		out.WriteString("[img-")
		out.Write(strconv.AppendInt(digits[:0], int64(id), 10))
		out.WriteByte(']')
	}
	for i := placed; i < count; i++ {
		writeTag(firstID + i)
	}
	rest = prompt
	for i := 0; i < placed; i++ {
		head, tail, _ := strings.Cut(rest, marker)
		out.WriteString(head)
		writeTag(firstID + i)
		rest = tail
	}
	out.WriteString(rest)
	return out.String()
}

// Some native Qwen templates reject even a second leading instruction. Other
// templates may implement developer roles or accept interleaved instructions.
func nativeInstructionMessages(m *Model, messages []api.Message) ([]api.Message, error) {
	if m == nil || !strings.HasPrefix(m.metadata.String("general.architecture"), "qwen") {
		return messages, nil
	}
	for i, message := range messages {
		if message.Role == "developer" || (i > 0 && message.Role == "system") {
			source := m.metadata.String("tokenizer.chat_template")
			if strings.Contains(source, "raise_exception('System message must be at the beginning.')") && !strings.Contains(source, "developer") {
				normalized, err := renderers.NormalizeQwenInstructionMessages(messages)
				if err != nil {
					return nil, api.StatusError{StatusCode: 400, ErrorMessage: err.Error()}
				}
				return normalized, nil
			}
			break
		}
	}
	return messages, nil
}

// Go templates issue many small writes for alternating turns. Reserve room
// before appending and return the builder's final string without copying it.
type promptWriter struct{ strings.Builder }

func (w *promptWriter) Write(p []byte) (int, error) {
	w.Grow(len(p))
	return w.Builder.Write(p)
}

func renderPrompt(m *Model, msgs []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error) {
	if m.Config.Renderer != "" {
		rendererName := resolveRendererName(m)
		rendered, err := renderers.RenderWithRenderer(rendererName, msgs, tools, think)
		if err != nil {
			return "", err
		}
		return rendered, nil
	}

	var b promptWriter
	thinkVal := false
	thinkLevel := ""
	if think != nil {
		thinkVal = think.Bool()
		thinkLevel = think.String()
	}
	if err := m.Template.Execute(&b, template.Values{Messages: msgs, Tools: tools, Think: thinkVal, ThinkLevel: thinkLevel, IsThinkSet: think != nil}); err != nil {
		return "", err
	}
	return b.String(), nil
}
