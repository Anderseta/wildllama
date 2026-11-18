package imagegen

import (
	"errors"
	"image"
	"image/png"
	"io"
	"os"
)

// decodeOutputPNG checks the dimensions and decodes the complete PNG stream.
// DecodeConfig alone accepts a valid header followed by a truncated image.
// Check the bounded requested dimensions first, then use the standard decoder
// to verify chunk checksums, compressed pixels and completion. No decoded image
// or decoder buffer is retained between jobs.
func decodeOutputPNG(path string, width, height int) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("image output is not a regular file")
	}
	config, format, err := image.DecodeConfig(f)
	if err != nil || format != "png" || config.Width != width || config.Height != height {
		return nil, errors.New("invalid PNG header or unexpected dimensions")
	}
	if width < 64 || width > 4096 || height < 64 || height > 4096 {
		return nil, errors.New("PNG dimensions exceed the image profile bounds")
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	decoded, err := png.Decode(f)
	if err != nil {
		return nil, err
	}
	after, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if after.Size() != info.Size() || !after.ModTime().Equal(info.ModTime()) {
		return nil, errors.New("PNG changed during validation")
	}
	return decoded, nil
}
