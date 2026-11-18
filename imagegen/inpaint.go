package imagegen

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"

	xdraw "golang.org/x/image/draw"
)

// White selects editable pixels. Coordinates span the first reference's full raster.
type InpaintMask struct {
	AssetID string `json:"asset_id"`
	PNG     string `json:"png"`
}

func (m *InpaintMask) bitmap() (*image.Gray, error) {
	if m.AssetID == "" || len(m.PNG) > 700000 {
		return nil, errors.New("invalid inpaint reference or mask size")
	}
	data, err := base64.StdEncoding.DecodeString(m.PNG)
	if err != nil {
		return nil, err
	}
	cfg, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil || cfg.Width > 1024 || cfg.Height > 1024 {
		return nil, errors.New("invalid inpaint PNG dimensions")
	}
	img, err := png.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	mask := image.NewGray(img.Bounds())
	draw.Draw(mask, mask.Bounds(), img, image.Point{}, draw.Src)
	selected := false
	for i, v := range mask.Pix {
		if v >= 128 {
			mask.Pix[i] = 255
			selected = true
		} else {
			mask.Pix[i] = 0
		}
	}
	if !selected {
		return nil, errors.New("marque uma área na imagem antes de gerar")
	}
	return mask, nil
}
func (m *InpaintMask) Validate() error { _, err := m.bitmap(); return err }
func (m *InpaintMask) writeMask(dir string, w, h int) error {
	mask, err := m.bitmap()
	if err != nil {
		return err
	}
	scaled := image.NewGray(image.Rect(0, 0, w, h))
	xdraw.NearestNeighbor.Scale(scaled, scaled.Bounds(), mask, mask.Bounds(), draw.Src, nil)
	f, err := os.Create(filepath.Join(dir, "mask.png"))
	if err != nil {
		return err
	}
	return errors.Join(png.Encode(f, scaled), f.Close())
}
func readInpaintImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}
func (m *InpaintMask) compose(ctx context.Context, reference, generated, output string) (image.Point, error) {
	var zero image.Point
	mask, err := m.bitmap()
	if err != nil {
		return zero, err
	}
	original, err := readInpaintImage(reference)
	if err != nil {
		return zero, err
	}
	edited, err := readInpaintImage(generated)
	if err != nil {
		return zero, err
	}
	b := original.Bounds()
	if err = CheckDiskSpace(ctx, output, int64(b.Dx())*int64(b.Dy())*8+(1<<20)); err != nil {
		return zero, err
	}
	// Start with the original raster to preserve transparency and 16-bit samples
	// exactly outside the mask. Resize only the generated pixels.
	var dst draw.Image
	switch src := original.(type) {
	case *image.NRGBA:
		dst = src
	case *image.NRGBA64:
		dst = src
	default:
		if original.ColorModel() == color.Gray16Model || original.ColorModel() == color.RGBA64Model {
			dst = image.NewNRGBA64(b)
		} else {
			dst = image.NewNRGBA(b)
		}
		draw.Draw(dst, b, original, b.Min, draw.Src)
	}
	if edited.Bounds() != b {
		scaled := image.NewNRGBA(b)
		xdraw.BiLinear.Scale(scaled, b, edited, edited.Bounds(), draw.Src, nil)
		edited = scaled
	}
	for y := 0; y < b.Dy(); y++ {
		if err = ctx.Err(); err != nil {
			return zero, err
		}
		my := (2*y + 1) * mask.Rect.Dy() / (2 * b.Dy())
		row := mask.Pix[my*mask.Stride:]
		start := 0
		for x := 0; x <= b.Dx(); x++ {
			if x < b.Dx() && row[(2*x+1)*mask.Rect.Dx()/(2*b.Dx())] != 0 {
				continue
			}
			if start < x {
				rect := image.Rect(start, y, x, y+1).Add(b.Min)
				draw.Draw(dst, rect, edited, rect.Min, draw.Src)
			}
			start = x + 1
		}
	}
	tmp, err := os.CreateTemp(filepath.Dir(output), ".inpaint-*.png")
	if err != nil {
		return zero, err
	}
	defer os.Remove(tmp.Name())
	if err = errors.Join(png.Encode(tmp, dst), tmp.Close()); err != nil {
		return zero, err
	}
	if err = ctx.Err(); err != nil {
		return zero, err
	}
	if err = os.Rename(tmp.Name(), output); err != nil {
		return zero, err
	}
	return b.Size(), nil
}
