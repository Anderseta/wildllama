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

	xdraw "golang.org/x/image/draw"
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
	bitmap, err := mask.bitmap()
	if err != nil {
		t.Fatal(err)
	}
	rendered, err := readInpaintImage(raw)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := composeInpaint(context.Background(), bitmap, source, rendered, output); err != nil {
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

func TestOverlayInpaintMatchesFullCanvas(t *testing.T) {
	for _, sizes := range [][6]int{{21, 15, 9, 7, 13, 6}, {13, 6, 17, 9, 13, 6}, {3, 4, 17, 15, 2, 2}, {8, 7, 1, 1, 15, 13}, {18, 11, 6, 7, 31, 4}} {
		w, h, mw, mh, sw, sh := sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5]
		b := image.Rect(3, 5, w+3, h+5)
		mask := image.NewGray(image.Rect(0, 0, mw, mh))
		for y := 0; y < mh; y++ {
			for x := 0; x < mw; x++ {
				if (x*3+y)%5 < 2 {
					mask.SetGray(x, y, color.Gray{255})
				}
			}
		}
		edited := image.NewNRGBA(image.Rect(0, 0, sw, sh))
		for y := 0; y < sh; y++ {
			for x := 0; x < sw; x++ {
				edited.SetNRGBA(x, y, color.NRGBA{uint8(17 * x), uint8(13 * y), 191, uint8(137 + x + y)})
			}
		}
		original := image.NewNRGBA64(b)
		draw.Draw(original, b, image.NewUniform(color.NRGBA64{R: 18023, G: 601, B: 299, A: 34987}), image.Point{}, draw.Src)
		want := image.NewNRGBA64(b)
		copy(want.Pix, original.Pix)
		full := image.NewNRGBA(b)
		xdraw.BiLinear.Scale(full, b, edited, edited.Bounds(), draw.Src, nil)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				if mask.GrayAt((2*x+1)*mw/(2*w), (2*y+1)*mh/(2*h)).Y == 255 {
					p := b.Min.Add(image.Pt(x, y))
					draw.Draw(want, image.Rectangle{Min: p, Max: p.Add(image.Pt(1, 1))}, full, p, draw.Src)
				}
			}
		}
		if err := overlayInpaint(t.Context(), original, edited, mask); err != nil || !bytes.Equal(want.Pix, original.Pix) {
			t.Fatalf("pixel mismatch for %v: %v", sizes, err)
		}
	}
}

func TestInpaintEnginePublishesOnlyCompleteEdit(t *testing.T) {
	for _, mode := range []string{"generate", "truncated-output", "late-failure"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			p := testProfile(t, dir)
			p.VisionEncoder = p.DiffusionModel
			original := image.NewNRGBA(image.Rect(0, 0, 96, 80))
			draw.Draw(original, original.Bounds(), image.White, image.Point{}, draw.Src)
			reference := filepath.Join(dir, "source.png")
			putModel(t, reference, encodedInpaintPNG(t, original))
			rawMask := image.NewGray(image.Rect(0, 0, 2, 1))
			rawMask.SetGray(1, 0, color.Gray{255})
			mask := InpaintMask{AssetID: "source", PNG: base64.StdEncoding.EncodeToString(encodedInpaintPNG(t, rawMask))}
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			outDir := filepath.Join(dir, "outputs")
			engine, err := NewEngine(Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: outDir})
			if err != nil {
				t.Fatal(err)
			}
			defer engine.Close()
			t.Setenv("WILDIMAGE_FAKE_CHILD", mode)
			output := filepath.Join(outDir, "final.png")
			result, err := engine.Run(t.Context(), p, Request{Prompt: "literal --prompt ; &", Seed: 17, OutputPath: output, Inpaint: &mask, ReferenceImages: []string{reference}}, nil)
			if mode != "generate" {
				if err == nil || result.Path != "" {
					t.Fatalf("published failed edit: %#v %v", result, err)
				}
				if _, err := os.Stat(output); !os.IsNotExist(err) {
					t.Fatalf("failed edit exists: %v", err)
				}
				return
			}
			if err != nil || result.Width != 96 || result.Height != 80 || result.Seed != 17 {
				t.Fatalf("result %#v %v", result, err)
			}
			img, err := readInpaintImage(result.Path)
			if err != nil {
				t.Fatal(err)
			}
			if color.NRGBAModel.Convert(img.At(10, 20)) != (color.NRGBA{255, 255, 255, 255}) || color.NRGBAModel.Convert(img.At(80, 20)) == (color.NRGBA{255, 255, 255, 255}) {
				t.Fatal("protected or edited area changed incorrectly")
			}
		})
	}
}
