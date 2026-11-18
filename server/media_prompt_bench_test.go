package server

import (
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func BenchmarkImageTaggedLongPrompt(b *testing.B) {
	body := strings.Repeat(strings.Repeat("context ", 512)+"[img]", 16)
	msg := []api.Message{{Content: body, Images: make([]api.ImageData, 24)}}
	b.ReportAllocs()
	for b.Loop() {
		messages, media, err := imageTaggedMessages(&Model{}, msg, true)
		if err != nil || len(media) != 24 || len(messages) != 1 {
			b.Fatal(err)
		}
	}
}

func TestImageTaggedPromptOrdering(t *testing.T) {
	model := &Model{}
	for _, test := range []struct {
		text, want string
		count      int
	}{
		{"sem marcador", "[img-9][img-10]sem marcador", 2},
		{"a[img]b", "[img-10][img-11]a[img-9]b", 3},
		{"[img][img] [img]", "[img-9][img-10] [img]", 2},
		{"[img-9]", "[img-9][img-9]", 1},
		{"", "[img-9]", 1},
	} {
		input := []api.Message{{Images: make([]api.ImageData, 9)}, {Content: test.text, Images: make([]api.ImageData, test.count)}}
		for _, clearMedia := range []bool{false, true} {
			got, media, err := imageTaggedMessages(model, input, clearMedia)
			if err != nil || got[1].Content != test.want || len(media) != 9+test.count || len(input[1].Images) != test.count || (got[1].Images == nil) != clearMedia || media[len(media)-1].ID != 8+test.count {
				t.Fatalf("%q: %#v, %v", test.text, got, err)
			}
		}
	}
	model.Config.Renderer = "native"
	input := []api.Message{{Content: "A [img] B", Images: []api.ImageData{{1, 2}}}}
	got, media, err := imageTaggedMessages(model, input, true)
	if err != nil || got[0].Content != input[0].Content || len(media) != 1 || !slices.Equal(media[0].Data, input[0].Images[0]) {
		t.Fatal(got, err)
	}
}
