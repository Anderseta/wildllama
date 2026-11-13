package imagegen

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRunnerReferencePrompts(t *testing.T) {
	names := []string{"video_patch_proj.weight", "model.diffusion_model.condition_proj.weight"}
	h3, _ := writeGGUFTensorFixture(t, names)
	video, _ := writeGGUFTensorFixture(t, append(names, "diffusion_model.audio_patch_proj.weight"))
	qwen, _ := writeGGUFTensorFixture(t, []string{"model.diffusion_model.txt_in.text_norm.weight"})
	header, err := json.Marshal(map[string]tensorDescriptor{
		names[0]: {DType: "F32", Shape: []int64{1}, Offsets: []int64{0, 4}},
		names[1]: {DType: "F32", Shape: []int64{1}, Offsets: []int64{4, 8}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for len(header)%8 != 0 {
		header = append(header, ' ')
	}
	data := append(binary.LittleEndian.AppendUint64(nil, uint64(len(header))), header...)
	safe := filepath.Join(t.TempDir(), "renamed.safetensors")
	if err := os.WriteFile(safe, append(data, make([]byte, 8)...), 0600); err != nil {
		t.Fatal(err)
	}

	const prompt = "Coloque <image1> no fundo de <image2>; preserve <Picture 1> e <image3>."
	const negative = "Não copie a sombra de <image2>."
	const adapted = "Coloque <Picture 1> no fundo de <Picture 2>; preserve <Picture 1> e <image3>."
	for _, test := range []struct {
		name, model, positive, negative string
		references                      []string
	}{
		{"MiniMax GGUF", h3, adapted, "Não copie a sombra de <Picture 2>.", []string{"one", "two"}},
		{"imported MiniMax", safe, adapted, "Não copie a sombra de <Picture 2>.", []string{"one", "two"}},
		{"Qwen", qwen, prompt, negative, []string{"one", "two"}},
		{"H3 video", video, prompt, negative, []string{"one", "two"}},
		{"no attachments", h3, prompt, negative, nil},
		{"unrecognized file", "missing.gguf", prompt, negative, []string{"one", "two"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, neg := runnerReferencePrompts(test.model, Request{
				Prompt: prompt, NegativePrompt: negative, ReferenceImages: test.references,
			})
			if got != test.positive || neg != test.negative {
				t.Fatalf("unexpected prompts: %q and %q", got, neg)
			}
		})
	}
}
