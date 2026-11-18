package imagegen

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func encodedInpaintPNG(t *testing.T, img image.Image) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}
func TestInpaintProtectedPixels(t *testing.T) {
	dir := t.TempDir()
	original := image.NewNRGBA64(image.Rect(0, 0, 9, 7))
	for y := 0; y < 7; y++ {
		for x := 0; x < 9; x++ {
			original.SetNRGBA64(x, y, color.NRGBA64{R: uint16(500 + x), G: 65500, B: 313, A: uint16(y * 9100)})
		}
	}
	edit := image.NewRGBA(image.Rect(0, 0, 3, 3))
	draw.Draw(edit, edit.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
	source := filepath.Join(dir, "original.png")
	raw := filepath.Join(dir, "edited.png")
	output := filepath.Join(dir, "out.png")
	for path, img := range map[string]image.Image{source: original, raw: edit} {
		if err := os.WriteFile(path, encodedInpaintPNG(t, img), 0600); err != nil {
			t.Fatal(err)
		}
	}
	m := image.NewGray(image.Rect(0, 0, 3, 1))
	mask := InpaintMask{AssetID: "source", PNG: base64.StdEncoding.EncodeToString(encodedInpaintPNG(t, m))}
	if err := mask.Validate(); err == nil {
		t.Fatal("empty mask accepted")
	}
	m.SetGray(1, 0, color.Gray{255})
	mask.PNG = base64.StdEncoding.EncodeToString(encodedInpaintPNG(t, m))
	if _, err := mask.compose(context.Background(), source, raw, output); err != nil {
		t.Fatal(err)
	}
	actual, err := readInpaintImage(output)
	if err != nil {
		t.Fatal(err)
	}
	if actual.Bounds() != original.Bounds() {
		t.Fatal("wrong output size")
	}
	for y := 0; y < 7; y++ {
		for x := 0; x < 9; x++ {
			want := original.NRGBA64At(x, y)
			if x >= 3 && x < 6 {
				want = color.NRGBA64{65535, 65535, 65535, 65535}
			}
			if actual.At(x, y) != want {
				t.Fatalf("pixel (%d,%d): %v want %v", x, y, actual.At(x, y), want)
			}
		}
	}
}
