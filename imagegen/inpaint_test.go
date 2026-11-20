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
	"slices"
	"strings"
	"testing"

	xdraw "golang.org/x/image/draw"
)

func TestCroppedInpaintCoordinates(t *testing.T) {
	original := image.Pt(1801, 1057)
	canvas := image.Pt(1024, 768)
	mask := image.NewGray(image.Rect(0, 0, 127, 99))
	selected := image.Rect(119, 89, 127, 98)
	draw.Draw(mask, selected, image.White, image.Point{}, draw.Src)
	mask.SetGray(121, 92, color.Gray{}) // A hole inside the selected bounds.
	crop, work := planInpaintCrop(mask, original, canvas)
	if crop.Max != original || crop.Size() != image.Pt(512, 512) || work != crop.Size() {
		t.Fatalf("unexpected crop %v and work canvas %v", crop, work)
	}
	dir := t.TempDir()
	if err := writeInpaintMask(mask, dir, work.X, work.Y, crop, original); err != nil {
		t.Fatal(err)
	}
	mapped, err := readInpaintImage(filepath.Join(dir, "mask.png"))
	if err != nil {
		t.Fatal(err)
	}
	b := image.Rectangle{Max: original}
	dst := image.NewNRGBA64(b)
	old := color.NRGBA64{R: 0x1234, G: 0x5678, B: 0x1357, A: 0x789a}
	for y := 0; y < original.Y; y++ {
		for x := 0; x < original.X; x++ {
			dst.SetNRGBA64(x, y, old)
		}
	}
	edit := image.NewNRGBA(image.Rectangle{Max: work})
	for y := 0; y < work.Y; y++ {
		for x := 0; x < work.X; x++ {
			edit.SetNRGBA(x, y, color.NRGBA{R: uint8(x), G: uint8(y), A: 255})
		}
	}
	if err := overlayInpaintRegion(t.Context(), dst, edit, mask, crop); err != nil {
		t.Fatal(err)
	}
	for y := 0; y < original.Y; y++ {
		for x := 0; x < original.X; x++ {
			mx, my := (2*x+1)*mask.Rect.Dx()/(2*original.X), (2*y+1)*mask.Rect.Dy()/(2*original.Y)
			inside := mask.GrayAt(mx, my).Y != 0
			want := old
			local := image.Pt(x, y).Sub(crop.Min)
			if inside {
				want = color.NRGBA64Model.Convert(edit.At(local.X, local.Y)).(color.NRGBA64)
			}
			if got := dst.NRGBA64At(x, y); got != want {
				t.Fatalf("pixel %d,%d = %v, want %v", x, y, got, want)
			}
			if image.Pt(x, y).In(crop) {
				v, _, _, _ := mapped.At(local.X, local.Y).RGBA()
				if (v != 0) != inside {
					t.Fatalf("work mask differs at %d,%d", x, y)
				}
			}
		}
	}
	draw.Draw(mask, mask.Bounds(), image.White, image.Point{}, draw.Src)
	if r, c := planInpaintCrop(mask, original, canvas); !r.Empty() || c != canvas {
		t.Fatalf("full mask changed canvas: %v %v", r, c)
	}
}

func TestInpaintReducedCropWithTwoReferences(t *testing.T) {
	original, canvas := image.Pt(1399, 997), image.Pt(384, 288)
	mask := image.NewGray(image.Rect(0, 0, 71, 63))
	draw.Draw(mask, image.Rect(25, 7, 48, 29), image.White, image.Point{}, draw.Src)
	mask.SetGray(4, 8, color.Gray{Y: 255})
	crop, work := planInpaintCrop(mask, original, canvas)
	if crop.Empty() || work.X >= crop.Dx() || work.X > canvas.X || work.Y > canvas.Y {
		t.Fatalf("unexpected crop %v, work %v", crop, work)
	}
	refs := []string{"first.png", "second.png"}
	request := Request{Inpaint: &InpaintMask{Crop: true}, ReferenceImages: refs, crop: crop}
	args := buildArguments(Profile{Width: work.X, Height: work.Y}, request, "job", "result.png", 1)
	value := func(name string) string {
		i := slices.Index(args, name)
		if i < 0 || i+1 >= len(args) {
			t.Fatalf("missing %s", name)
		}
		return args[i+1]
	}
	if value("--init-img") != refs[0] || value("--inpaint-crop") == "" || !strings.Contains(strings.Join(args, " "), "-r first.png -r second.png") {
		t.Fatal("runner lost crop or reference order", args)
	}
	edit := image.NewRGBA(image.Rectangle{Max: work})
	for y := 0; y < work.Y; y++ {
		for x := 0; x < work.X; x++ {
			edit.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), A: 255})
		}
	}
	dst := image.NewNRGBA(image.Rectangle{Max: original})
	if err := overlayInpaintRegion(t.Context(), dst, edit, mask, crop); err != nil {
		t.Fatal(err)
	}
	expected := image.NewNRGBA(crop)
	xdraw.BiLinear.Scale(expected, crop, edit, edit.Bounds(), draw.Src, nil)
	for y := 0; y < original.Y; y++ {
		for x := 0; x < original.X; x++ {
			mx, my := (2*x+1)*mask.Rect.Dx()/(2*original.X), (2*y+1)*mask.Rect.Dy()/(2*original.Y)
			want := color.NRGBA{}
			if mask.GrayAt(mx, my).Y != 0 {
				want = expected.NRGBAAt(x, y)
			}
			if got := dst.NRGBAAt(x, y); got != want {
				t.Fatalf("pixel %d,%d = %v, want %v", x, y, got, want)
			}
		}
	}
}

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
	if _, err := composeInpaint(context.Background(), bitmap, source, rendered, output, image.Rectangle{}); err != nil {
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
