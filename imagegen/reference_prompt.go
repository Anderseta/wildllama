package imagegen

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/fs/safetensors"
)

var imageReferenceTag = regexp.MustCompile(`<image([1-9][0-9]*)>`)

// Reference syntax adapters. Add one entry for each architecture that needs
// a spelling other than <imageN>. Match by tensor names, which also works for
// imported and renamed models. Qwen Image 2.1 and unknown models pass through.
type imageReferenceAdapter struct {
	family   string
	format   string
	required []string
	excluded []string
}

var imageReferenceAdapters = []imageReferenceAdapter{
	{ // MiniMax H3 Image; the video model has an additional audio projection.
		family:   "minimax-h3-image",
		format:   "<Picture %d>",
		required: []string{"video_patch_proj.weight", "condition_proj.weight"},
		excluded: []string{"audio_patch_proj.weight"},
	},
}

// Keep the canonical form in the request and history. Only adapt the text
// passed to a known model that needs a different spelling.
func runnerReferencePrompts(modelPath string, r Request) (string, string) {
	count := len(r.ReferenceImages)
	if count == 0 || (!imageReferenceTag.MatchString(r.Prompt) && !imageReferenceTag.MatchString(r.NegativePrompt)) {
		return r.Prompt, r.NegativePrompt
	}
	format := imageReferenceFormat(modelPath)
	if format == "" {
		return r.Prompt, r.NegativePrompt
	}
	replace := func(tag string) string {
		number, err := strconv.Atoi(tag[len("<image") : len(tag)-1])
		if err != nil || number > count {
			return tag
		}
		return fmt.Sprintf(format, number)
	}
	return imageReferenceTag.ReplaceAllStringFunc(r.Prompt, replace),
		imageReferenceTag.ReplaceAllStringFunc(r.NegativePrompt, replace)
}

// Read model headers only, never weight payloads.
func imageReferenceFormat(path string) string {
	if adapter := detectImageReferenceAdapter(path); adapter != nil {
		return adapter.format
	}
	return ""
}

func detectImageReferenceAdapter(path string) *imageReferenceAdapter {
	names := make(map[string]bool)
	observe := func(name string) {
		name = strings.TrimPrefix(strings.TrimPrefix(name, "model."), "diffusion_model.")
		names[name] = true
	}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".gguf":
		metadata, err := gguf.ReadFileMetadata(path, 32)
		if err != nil {
			return nil
		}
		for _, tensor := range metadata.TensorInfos() {
			observe(tensor.Name)
		}
	case ".safetensors":
		tensors, err := safetensors.OpenForExtraction(path)
		if err != nil {
			return nil
		}
		defer tensors.Close()
		for _, name := range tensors.ListTensors() {
			observe(name)
		}
	}
adapters:
	for i := range imageReferenceAdapters {
		adapter := &imageReferenceAdapters[i]
		for _, name := range adapter.required {
			if !names[name] {
				continue adapters
			}
		}
		for _, name := range adapter.excluded {
			if names[name] {
				continue adapters
			}
		}
		return adapter
	}
	return nil
}
