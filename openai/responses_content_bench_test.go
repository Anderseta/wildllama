package openai

import (
	"fmt"
	"strings"
	"testing"
)

func TestResponsesMultipartOrderAndErrors(t *testing.T) {
	parts := []ResponsesContent{
		ResponsesTextContent{Text: "início "}, ResponsesImageContent{ImageURL: "data:image/png;base64,AQ=="},
		ResponsesOutputTextContent{Text: "meio "}, ResponsesImageContent{},
		ResponsesEncryptedContent{EncryptedContent: "fim"}, ResponsesImageContent{ImageURL: "data:image/png;base64,Ag=="},
	}
	text, images, err := convertResponsesContent(parts)
	if err != nil || text != "início meio fim" || len(images) != 2 || string(images[0]) != "\x01" || string(images[1]) != "\x02" {
		t.Fatalf("unexpected content %q, %v, %v", text, images, err)
	}
	for _, invalid := range []ResponsesContent{ResponsesFileContent{}, ResponsesImageContent{ImageURL: "data:image/png;base64,?"}} {
		text, images, err := convertResponsesContent(append(parts, invalid))
		if err == nil || text != "" || images != nil {
			t.Fatalf("partial result for invalid content: %q, %v, %v", text, images, err)
		}
	}
}

func BenchmarkResponsesMultipartText(b *testing.B) {
	for _, count := range []int{1, 32, 512} {
		b.Run(fmt.Sprint(count), func(b *testing.B) {
			part := strings.Repeat("olá mundo ", 12)
			in := make([]ResponsesContent, count)
			for i := range in {
				in[i] = ResponsesTextContent{Text: part}
			}
			b.ReportAllocs()
			for b.Loop() {
				text, _, err := convertResponsesContent(in)
				if err != nil || len(text) != len(part)*count {
					b.Fatal("invalid output", err)
				}
			}
		})
	}
}
