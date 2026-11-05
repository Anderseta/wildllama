package imagegen

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestReferenceImagesReachIsolatedRunnerInOrder(t *testing.T) {
	for _, count := range []int{1, 4} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			dir := t.TempDir()
			profile := testProfile(t, dir)
			profile.VisionEncoder = profile.DiffusionModel
			profile.PreserveReferenceSize = true
			refs := make([]string, count)
			before := make([]string, count)
			for i := range refs {
				refs[i] = filepath.Join(dir, fmt.Sprintf("reference-%d.png", count-i))
				referenceSizeHeader(t, refs[i], 96+i*32, 80+i*32)
				data, err := os.ReadFile(refs[i])
				if err != nil {
					t.Fatal(err)
				}
				before[i] = digest(data)
			}
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			engine, err := NewEngine(Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: filepath.Join(dir, "outputs")})
			if err != nil {
				t.Fatal(err)
			}
			defer engine.Close()
			argsPath := filepath.Join(dir, "args.json")
			t.Setenv("WILDIMAGE_FAKE_CHILD", "generate")
			t.Setenv("WILDIMAGE_FAKE_ARGS", argsPath)
			result, err := engine.Run(t.Context(), profile, Request{Prompt: "literal --prompt ; &", Width: 128, Height: 160, ReferenceImages: refs, OutputPath: filepath.Join(dir, "outputs", "image.png")}, nil)
			if err != nil || result.Width != 96 || result.Height != 96 {
				t.Fatalf("first reference must select the canvas for one or four inputs: %#v, %v", result, err)
			}
			encoded, err := os.ReadFile(argsPath)
			if err != nil {
				t.Fatal(err)
			}
			var args, dispatched []string
			if err := json.Unmarshal(encoded, &args); err != nil {
				t.Fatal(err)
			}
			for i := 0; i+1 < len(args); i++ {
				if args[i] == "-r" {
					dispatched = append(dispatched, args[i+1])
				}
			}
			if !slices.Equal(dispatched, refs) {
				t.Fatalf("reference ordering changed: got %v, want %v", dispatched, refs)
			}
			for i, path := range refs {
				data, err := os.ReadFile(path)
				if err != nil || digest(data) != before[i] {
					t.Fatal("the original reference was modified", path, err)
				}
			}
			if err := ValidateRequest(profile, Request{Prompt: "fixture", ReferenceImages: []string{refs[0], refs[0], refs[0], refs[0], refs[0]}}); err == nil {
				t.Fatal("accepted more than four reference images")
			}
		})
	}
}

func TestConditioningCacheReferenceOrderAndFourthInput(t *testing.T) {
	dir := t.TempDir()
	profile := testProfile(t, dir)
	profile.VisionEncoder = profile.DiffusionModel
	config := Config{WorkDir: filepath.Join(dir, "work"), ConditioningCacheMiB: 256}
	refs := make([]string, 4)
	for i := range refs {
		refs[i] = filepath.Join(dir, fmt.Sprintf("reference-%d.png", i))
		referenceSizeHeader(t, refs[i], 64+i*32, 64)
	}
	key := func(refs []string) string {
		t.Helper()
		env, err := conditioningCacheEnvironment(context.Background(), config, profile, Request{ReferenceImages: refs}, "fixture")
		if err != nil {
			t.Fatal(err)
		}
		return envValue(env, "WILDIMAGE_CONDITION_NAMESPACE")
	}
	first := key(refs)
	if key(refs) != first || key(refs[:1]) == first {
		t.Fatal("reference count did not participate in the cache identity")
	}
	reordered := slices.Clone(refs)
	reordered[0], reordered[3] = reordered[3], reordered[0]
	if key(reordered) == first {
		t.Fatal("reference order did not participate in the cache identity")
	}
	referenceSizeHeader(t, refs[3], 192, 96)
	if key(refs) == first {
		t.Fatal("the fourth reference did not participate in the cache identity")
	}
}
