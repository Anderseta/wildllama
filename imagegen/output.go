package imagegen

import (
	"errors"
	"image"
	"image/png"
	"io"
	"os"
)

// validateOutputPNG checks the dimensions and decodes the complete PNG stream.
// DecodeConfig alone accepts a valid header followed by a truncated image.
// Check the bounded requested dimensions first, then use the standard decoder
// to verify chunk checksums, compressed pixels and completion. No decoded image
// or decoder buffer is retained between jobs.
func validateOutputPNG(path string, width, height int) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return errors.New("image output is not a regular file")
	}
	config, format, err := image.DecodeConfig(f)
	if err != nil || format != "png" || config.Width != width || config.Height != height {
		return errors.New("invalid PNG header or unexpected dimensions")
	}
	if width < 64 || width > 4096 || height < 64 || height > 4096 {
		return errors.New("PNG dimensions exceed the image profile bounds")
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if _, err := png.Decode(f); err != nil {
		return err
	}
	after, err := f.Stat()
	if err != nil {
		return err
	}
	if after.Size() != info.Size() || !after.ModTime().Equal(info.ModTime()) {
		return errors.New("PNG changed during validation")
	}
	return nil
}
