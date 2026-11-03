package imagegen

import (
	"errors"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
)

// ResolveProfileDimensions keeps the original reference untouched. The native
// pipeline resamples its working tensors to these supported canvas dimensions.
// An explicit reference-size preference takes priority over request overrides.
func ResolveProfileDimensions(p Profile, r Request) (Profile, error) {
	if p.PreserveReferenceSize && len(r.ReferenceImages) > 0 {
		config, err := referenceImageConfig(r.ReferenceImages[0])
		if err != nil {
			return p, err
		}
		scale := math.Min(1, 4096/float64(max(config.Width, config.Height)))
		nearest := func(value int) int {
			return max(64, min(4096, int(math.Round(float64(value)*scale/32))*32))
		}
		p.Width, p.Height = nearest(config.Width), nearest(config.Height)
		return p, nil
	}
	if r.Width != 0 {
		p.Width = r.Width
	}
	if r.Height != 0 {
		p.Height = r.Height
	}
	return p, nil
}

func referenceImageConfig(path string) (image.Config, error) {
	if !filepath.IsAbs(path) {
		return image.Config{}, errors.New("reference image paths must be absolute")
	}
	f, err := os.Open(path)
	if err != nil {
		return image.Config{}, fmt.Errorf("reference image: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() || info.Size() > 48<<20 {
		return image.Config{}, errors.New("reference image must be a regular image of at most 48 MiB")
	}
	config, format, err := image.DecodeConfig(io.LimitReader(f, 48<<20))
	if err != nil || (format != "png" && format != "jpeg" && format != "webp") || config.Width <= 0 || config.Height <= 0 || int64(config.Width)*int64(config.Height) > 64<<20 {
		return image.Config{}, errors.New("reference image must be a valid PNG, JPEG or WebP with at most 64 megapixels")
	}
	return config, nil
}
