package imagegen

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"hash/crc32"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

// Dimension-only fixtures avoid allocating large images for metadata tests.
func referenceSizeHeader(t *testing.T, path string, width, height int) {
	t.Helper()
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	header := bytes.Clone(buffer.Bytes()[:33])
	binary.BigEndian.PutUint32(header[16:20], uint32(width))
	binary.BigEndian.PutUint32(header[20:24], uint32(height))
	binary.BigEndian.PutUint32(header[29:33], crc32.ChecksumIEEE(header[12:29]))
	putModel(t, path, header)
}

func TestResolveReferenceSizeNearestSupportedCanvas(t *testing.T) {
	for _, test := range []struct {
		name                string
		width, height, w, h int
	}{
		{"aligned", 1920, 1088, 1920, 1088},
		{"nearest", 1920, 1080, 1920, 1088},
		{"minimum", 17, 31, 64, 64},
		{"landscape limit", 8192, 4096, 4096, 2048},
		{"portrait limit", 6000, 9000, 2720, 4096},
		{"extreme aspect minimum", 1, 4097, 64, 4096},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "reference.png")
			referenceSizeHeader(t, path, test.width, test.height)
			before, _ := os.ReadFile(path)
			profile := DefaultProfile("fixture")
			profile.PreserveReferenceSize = true
			actual, err := ResolveProfileDimensions(profile, Request{Width: 3, Height: 9999, ReferenceImages: []string{path}})
			if err != nil || actual.Width != test.w || actual.Height != test.h {
				t.Fatalf("effective size %dx%d, error %v", actual.Width, actual.Height, err)
			}
			after, _ := os.ReadFile(path)
			if digest(before) != digest(after) || profile.Width != 1024 || profile.Height != 1024 {
				t.Fatal("reference or saved profile changed")
			}
		})
	}
}

func TestReferenceSizePreferenceScopeAndValidation(t *testing.T) {
	dir := t.TempDir()
	first, second := filepath.Join(dir, "first.png"), filepath.Join(dir, "second.png")
	referenceSizeHeader(t, first, 96, 80)
	referenceSizeHeader(t, second, 160, 128)
	profile := testProfile(t, dir)
	profile.VisionEncoder = profile.DiffusionModel
	for _, flag := range []bool{false, true} {
		profile.PreserveReferenceSize = flag
		for _, refs := range [][]string{nil, {first, second}} {
			request := Request{Prompt: "fixture", Width: 128, Height: 160, ReferenceImages: refs}
			actual, err := ResolveProfileDimensions(profile, request)
			if err != nil {
				t.Fatal(err)
			}
			w, h := 128, 160
			if flag && len(refs) > 0 {
				w, h = 96, 96
			}
			if actual.Width != w || actual.Height != h || ValidateRequest(profile, request) != nil {
				t.Fatalf("flag=%v references=%d: %#v", flag, len(refs), actual)
			}
		}
	}
	profile.PreserveReferenceSize = true
	if err := ValidateRequest(profile, Request{Prompt: "fixture", Width: 31}); err == nil {
		t.Fatal("reference preference changed no-reference validation")
	}
	referenceSizeHeader(t, first, 8192, 8193)
	if _, err := ResolveProfileDimensions(profile, Request{ReferenceImages: []string{first}}); err == nil {
		t.Fatal("accepted a reference beyond the existing 64 megapixel limit")
	}
	if _, err := ResolveProfileDimensions(profile, Request{ReferenceImages: []string{"relative.png"}}); err == nil {
		t.Fatal("accepted an arbitrary relative reference path")
	}
}

func TestReferenceSizeProfileBooleanAlwaysPersists(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		profile := DefaultProfile("fixture")
		profile.PreserveReferenceSize = enabled
		encoded, err := json.Marshal(profile)
		if err != nil || !bytes.Contains(encoded, []byte(`"preserve_reference_size":`)) {
			t.Fatal("reference preference capability missing from JSON", string(encoded), err)
		}
		var restored Profile
		if err := json.Unmarshal(encoded, &restored); err != nil || restored.PreserveReferenceSize != enabled {
			t.Fatal("reference preference did not survive JSON", err)
		}
	}
	var old Profile
	if err := json.Unmarshal([]byte(`{"width":64,"height":64}`), &old); err != nil || old.PreserveReferenceSize {
		t.Fatal("legacy profiles changed behavior", err)
	}
}

func TestReferenceSizeIsolatedRunnerUsesOriginalAndEffectiveDimensions(t *testing.T) {
	for _, mode := range []string{"generate", "late-failure", "saved-wait"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			profile := testProfile(t, dir)
			profile.VisionEncoder, profile.PreserveReferenceSize = profile.DiffusionModel, true
			path := filepath.Join(dir, "reference.png")
			var original bytes.Buffer
			if err := png.Encode(&original, image.NewRGBA(image.Rect(0, 0, 96, 80))); err != nil {
				t.Fatal(err)
			}
			putModel(t, path, original.Bytes())
			second := filepath.Join(dir, "second.png")
			putModel(t, second, original.Bytes())
			exe, _ := os.Executable()
			engine, err := NewEngine(Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: filepath.Join(dir, "outputs")})
			if err != nil {
				t.Fatal(err)
			}
			defer engine.Close()
			argsPath := filepath.Join(dir, "args.json")
			t.Setenv("WILDIMAGE_FAKE_CHILD", mode)
			t.Setenv("WILDIMAGE_FAKE_ARGS", argsPath)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			result, err := engine.Run(ctx, profile, Request{Prompt: "literal --prompt ; &", Width: 4096, Height: 31, Seed: 4, ReferenceImages: []string{path, second}, OutputPath: filepath.Join(dir, "outputs", "image.png")}, func(progress Progress) {
				if mode == "saved-wait" && progress.Phase == "saving" {
					cancel()
				}
			})
			if (err != nil) != (mode != "generate") || result.Width != 96 || result.Height != 96 {
				t.Fatalf("result=%#v error=%v", result, err)
			}
			if err := validateOutputPNG(result.Path, 96, 96); err != nil {
				t.Fatal(err)
			}
			encoded, _ := os.ReadFile(argsPath)
			var args []string
			if err := json.Unmarshal(encoded, &args); err != nil {
				t.Fatal(err)
			}
			var refs []string
			for i := 0; i+1 < len(args); i++ {
				if args[i] == "-r" {
					refs = append(refs, args[i+1])
				}
			}
			after, _ := os.ReadFile(path)
			secondAfter, _ := os.ReadFile(second)
			if len(refs) != 2 || refs[0] != path || refs[1] != second || digest(after) != digest(original.Bytes()) || digest(secondAfter) != digest(original.Bytes()) {
				t.Fatal("runner received a replaced/resized reference or original bytes changed", refs)
			}
			if mode == "saved-wait" && !errors.Is(err, context.Canceled) {
				t.Fatal("late cancellation error changed", err)
			}
		})
	}
}
