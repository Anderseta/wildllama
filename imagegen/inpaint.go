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
	"math"
	"os"
	"path/filepath"

	xdraw "golang.org/x/image/draw"
)

// White selects editable pixels. Coordinates span the first reference's full raster.
type InpaintMask struct {
	AssetID string `json:"asset_id"`
	PNG     string `json:"png"`
	Crop    bool   `json:"crop,omitempty"`
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
func selectedMaskBounds(mask *image.Gray) image.Rectangle {
	var selected image.Rectangle
	for y := 0; y < mask.Rect.Dy(); y++ {
		row := mask.Pix[y*mask.Stride : y*mask.Stride+mask.Rect.Dx()]
		if x := bytes.IndexByte(row, 255); x >= 0 {
			selected = selected.Union(image.Rect(x, y, bytes.LastIndexByte(row, 255)+1, y+1))
		}
	}
	return selected
}

func maskRasterRect(cells image.Rectangle, mask, original image.Point) image.Rectangle {
	return image.Rect(maskEdge(cells.Min.X, mask.X, original.X), maskEdge(cells.Min.Y, mask.Y, original.Y),
		maskEdge(cells.Max.X, mask.X, original.X), maskEdge(cells.Max.Y, mask.Y, original.Y))
}

func planInpaintCrop(mask *image.Gray, original, canvas image.Point) (image.Rectangle, image.Point) {
	selection := maskRasterRect(selectedMaskBounds(mask), mask.Rect.Size(), original)
	if selection.Empty() {
		return image.Rectangle{}, canvas
	}
	axis := func(lo, hi, size int) (int, int) {
		span := hi - lo
		length := min(size, max(512, span+2*max(64, span/4)))
		start := max(0, min(size-length, lo-(length-span)/2))
		return start, start + length
	}
	x0, x1 := axis(selection.Min.X, selection.Max.X, original.X)
	y0, y1 := axis(selection.Min.Y, selection.Max.Y, original.Y)
	crop := image.Rect(x0, y0, x1, y1)
	if crop.Size() == original {
		return image.Rectangle{}, canvas
	}
	scale := min(1.0, float64(canvas.X)/float64(crop.Dx()), float64(canvas.Y)/float64(crop.Dy()))
	align := func(n, limit int) int { return min(limit, max(64, 32*int(math.Round(float64(n)*scale/32)))) }
	return crop, image.Pt(align(crop.Dx(), canvas.X), align(crop.Dy(), canvas.Y))
}

func writeInpaintMask(mask *image.Gray, dir string, w, h int, crop image.Rectangle, original image.Point) error {
	scaled := image.NewGray(image.Rect(0, 0, w, h))
	if crop.Empty() {
		xdraw.NearestNeighbor.Scale(scaled, scaled.Bounds(), mask, mask.Bounds(), draw.Src, nil)
	} else {
		// Cover even a one-pixel mark when reducing the work canvas. Final
		// compositing still uses the original mask at original resolution.
		for y := 0; y < mask.Rect.Dy(); y++ {
			row := mask.Pix[y*mask.Stride : y*mask.Stride+mask.Rect.Dx()]
			for x := 0; x < len(row); {
				first := bytes.IndexByte(row[x:], 255)
				if first < 0 {
					break
				}
				x += first
				end := len(row)
				if gap := bytes.IndexByte(row[x:], 0); gap >= 0 {
					end = x + gap
				}
				r := maskRasterRect(image.Rect(x, y, end, y+1), mask.Rect.Size(), original).Intersect(crop)
				if !r.Empty() {
					r = r.Sub(crop.Min)
					xBasis, yBasis := int64(crop.Dx()), int64(crop.Dy())
					rect := image.Rect(int(int64(r.Min.X)*int64(w)/xBasis), int(int64(r.Min.Y)*int64(h)/yBasis),
						int((int64(r.Max.X)*int64(w)+xBasis-1)/xBasis), int((int64(r.Max.Y)*int64(h)+yBasis-1)/yBasis))
					draw.Draw(scaled, rect, image.White, image.Point{}, draw.Src)
				}
				x = end
			}
		}
	}
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

func composeInpaint(ctx context.Context, mask *image.Gray, reference string, edited image.Image, output string, crop image.Rectangle) (image.Point, error) {
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
	if err := overlayInpaintRegion(ctx, dst, edited, mask, crop); err != nil {
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
	return overlayInpaintRegion(ctx, dst, edited, mask, image.Rectangle{})
}

func overlayInpaintRegion(ctx context.Context, dst draw.Image, edited image.Image, mask *image.Gray, crop image.Rectangle) error {
	b := dst.Bounds()
	mw, mh := mask.Bounds().Dx(), mask.Bounds().Dy()
	region := image.Rectangle{}
	if !crop.Empty() || edited.Bounds() != b {
		region = selectedMaskBounds(mask)
		if err := ctx.Err(); err != nil {
			return err
		}
		rect := image.Rect(maskEdge(region.Min.X, mw, b.Dx()), maskEdge(region.Min.Y, mh, b.Dy()), maskEdge(region.Max.X, mw, b.Dx()), maskEdge(region.Max.Y, mh, b.Dy())).Add(b.Min)
		if rect.Empty() {
			return nil
		}
		if crop.Empty() {
			crop = b
		}
		if !rect.In(crop) || !crop.In(b) {
			return errors.New("inpaint crop no longer matches the reference")
		}
		scaled := image.NewNRGBA(rect)
		xdraw.BiLinear.Scale(scaled, crop, edited, edited.Bounds(), draw.Src, nil)
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
