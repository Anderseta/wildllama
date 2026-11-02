package imagegen

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestOutputPNGRejectsTruncationAndCorruption(t *testing.T) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 64, 64))); err != nil {
		t.Fatal(err)
	}
	complete := encoded.Bytes()
	corrupted := bytes.Clone(complete)
	idat := bytes.Index(corrupted, []byte("IDAT"))
	if idat < 0 {
		t.Fatal("test PNG has no pixel data")
	}
	corrupted[idat+4] ^= 1
	invalidPixels := bytes.Clone(corrupted)
	idatLength := int(binary.BigEndian.Uint32(invalidPixels[idat-4 : idat]))
	checksumOffset := idat + 4 + idatLength
	binary.BigEndian.PutUint32(invalidPixels[checksumOffset:checksumOffset+4], crc32.ChecksumIEEE(invalidPixels[idat:checksumOffset]))
	emptyPixels := append(bytes.Clone(complete[:33]), 0, 0, 0, 0, 'I', 'D', 'A', 'T')
	var emptyCRC [4]byte
	binary.BigEndian.PutUint32(emptyCRC[:], crc32.ChecksumIEEE([]byte("IDAT")))
	emptyPixels = append(emptyPixels, emptyCRC[:]...)
	emptyPixels = append(emptyPixels, complete[len(complete)-12:]...)
	for _, test := range []struct {
		name  string
		data  []byte
		width int
		valid bool
	}{
		{"complete", complete, 64, true},
		{"header only", complete[:33], 64, false},
		{"missing end", complete[:len(complete)-12], 64, false},
		{"corrupted pixel data", corrupted, 64, false},
		{"invalid compressed pixels with valid checksum", invalidPixels, 64, false},
		{"empty pixel stream with valid checksum", emptyPixels, 64, false},
		{"wrong dimensions", complete, 128, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "output.png")
			putModel(t, path, test.data)
			if err := validateOutputPNG(path, test.width, 64); (err == nil) != test.valid {
				t.Fatalf("valid=%v, error=%v", test.valid, err)
			}
		})
	}
}

func TestIsolatedProcessPreservesCompleteOutputOnLateFailure(t *testing.T) {
	for _, mode := range []string{"late-failure", "saved-wait", "truncated-output"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			profile := testProfile(t, dir)
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			engine, err := NewEngine(Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: filepath.Join(dir, "outputs")})
			if err != nil {
				t.Fatal(err)
			}
			defer engine.Close()
			t.Setenv("WILDIMAGE_FAKE_CHILD", mode)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			output := filepath.Join(dir, "outputs", "result.png")
			request := Request{Prompt: "literal --prompt ; & $(do not execute)", Seed: 42, OutputPath: output}
			result, err := engine.Run(ctx, profile, request, func(progress Progress) {
				if mode == "saved-wait" && progress.Phase == "saving" {
					cancel()
				}
			})
			if err == nil {
				t.Fatal("late runner failure/cancellation or incomplete PNG was hidden")
			}
			if mode == "truncated-output" {
				if result.Path != "" || !strings.Contains(err.Error(), "complete PNG") {
					t.Fatalf("incomplete output was published: %#v, %v", result, err)
				}
				if _, statErr := os.Stat(output); !errors.Is(statErr, os.ErrNotExist) {
					t.Fatal("incomplete output was not removed", statErr)
				}
				return
			}
			if mode == "saved-wait" && !errors.Is(err, context.Canceled) {
				t.Fatal("cancellation error was lost", err)
			}
			if result.Path != output || result.Width != 64 || result.Height != 64 {
				t.Fatalf("saved result was not returned: %#v, %v", result, err)
			}
			if err := validateOutputPNG(output, 64, 64); err != nil {
				t.Fatal("finished image was not preserved", err)
			}
			if engine.Status().Running {
				t.Fatal("image process remains running")
			}
		})
	}
}
