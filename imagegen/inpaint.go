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
func writeInpaintMask(mask *image.Gray, dir string, w, h int) error {
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

// maskEdge returns the first output pixel whose center maps to this mask
// column (or row). A half-pixel tie belongs to the following mask cell.
func maskEdge(cell, cells, pixels int) int {
	return int((2*int64(cell)*int64(pixels) + int64(cells) - 1) / (2 * int64(cells)))
}

func composeInpaint(ctx context.Context, mask *image.Gray, reference string, edited image.Image, output string) (image.Point, error) {
	var zero image.Point
	original, err := readInpaintImage(reference)
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
	if err := overlayInpaint(ctx, dst, edited, mask); err != nil {
		return zero, err
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

// Keep the full bilinear coordinate transform. Restrict the destination
// allocation and writes to the bounding rectangle of the selected mask cells.
func overlayInpaint(ctx context.Context, dst draw.Image, edited image.Image, mask *image.Gray) error {
	b := dst.Bounds()
	mw, mh := mask.Bounds().Dx(), mask.Bounds().Dy()
	region := image.Rectangle{}
	if edited.Bounds() != b {
		for y := 0; y < mh; y++ {
			row := mask.Pix[y*mask.Stride : y*mask.Stride+mw]
			x := bytes.IndexByte(row, 255)
			if x >= 0 {
				region = region.Union(image.Rect(x, y, bytes.LastIndexByte(row, 255)+1, y+1))
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		rect := image.Rect(maskEdge(region.Min.X, mw, b.Dx()), maskEdge(region.Min.Y, mh, b.Dy()), maskEdge(region.Max.X, mw, b.Dx()), maskEdge(region.Max.Y, mh, b.Dy())).Add(b.Min)
		if rect.Empty() {
			return nil
		}
		scaled := image.NewNRGBA(rect)
		xdraw.BiLinear.Scale(scaled, b, edited, edited.Bounds(), draw.Src, nil)
		edited = scaled
	} else {
		region = image.Rect(0, 0, mw, mh)
	}
	for y := region.Min.Y; y < region.Max.Y; y++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		y0, y1 := maskEdge(y, mh, b.Dy()), maskEdge(y+1, mh, b.Dy())
		if y0 == y1 {
			continue
		}
		row := mask.Pix[y*mask.Stride : y*mask.Stride+region.Max.X]
		for x := region.Min.X; x < len(row); {
			offset := bytes.IndexByte(row[x:], 255)
			if offset < 0 {
				break
			}
			x += offset
			end := len(row)
			if gap := bytes.IndexByte(row[x:], 0); gap >= 0 {
				end = x + gap
			}
			rect := image.Rect(maskEdge(x, mw, b.Dx()), y0, maskEdge(end, mw, b.Dx()), y1).Add(b.Min)
			draw.Draw(dst, rect, edited, rect.Min, draw.Src)
			x = end
		}
	}
	return ctx.Err()
}
