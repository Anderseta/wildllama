package imagegen

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

// Measure the edit's lossless host-side finalization, including PNG validation
// and publication, with both full-size and localized resized edits.
func BenchmarkFinalizeInpaint(b *testing.B) {
	for _, c := range []struct {
		name               string
		w, h, editW, editH int
	}{
		{"same-size", 1024, 1024, 1024, 1024}, {"localized-resize", 1536, 1024, 1024, 768},
	} {
		b.Run(c.name, func(b *testing.B) {
			dir := b.TempDir()
			source, generated, out := filepath.Join(dir, "source.png"), filepath.Join(dir, "generated.png"), filepath.Join(dir, "out.png")
			for _, item := range []struct {
				path string
				w, h int
			}{{source, c.w, c.h}, {generated, c.editW, c.editH}} {
				img := image.NewNRGBA(image.Rect(0, 0, item.w, item.h))
				for y := 0; y < item.h; y++ {
					for x := 0; x < item.w; x++ {
						img.SetNRGBA(x, y, color.NRGBA{uint8(x + y), uint8(2*y + x), uint8(x * y), 255})
					}
				}
				f, err := os.Create(item.path)
				if err != nil {
					b.Fatal(err)
				}
				if err := png.Encode(f, img); err != nil {
					b.Fatal(err)
				}
				if err := f.Close(); err != nil {
					b.Fatal(err)
				}
			}
			raw := image.NewGray(image.Rect(0, 0, 512, 512))
			draw.Draw(raw, image.Rect(83, 111, 130, 197), image.White, image.Point{}, draw.Src)
			var buf bytes.Buffer
			if err := png.Encode(&buf, raw); err != nil {
				b.Fatal(err)
			}
			mask := InpaintMask{AssetID: "fixture", PNG: base64.StdEncoding.EncodeToString(buf.Bytes())}
			b.ReportAllocs()
			for b.Loop() {
				decoded, err := decodeOutputPNG(generated, c.editW, c.editH)
				if err != nil {
					b.Fatal(err)
				}
				bitmap, err := mask.bitmap()
				if err != nil {
					b.Fatal(err)
				}
				if _, err := composeInpaint(b.Context(), bitmap, source, decoded, out); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
