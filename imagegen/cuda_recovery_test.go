package imagegen

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
)

func TestImageCUDAInitializationRecovery(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("CUDA eager recovery is specific to Windows")
	}
	for _, scenario := range []struct {
		name, mode, env string
		attempts        int
		saved, fails    bool
	}{
		{"after RAM offload", "cuda-module-after-offload", "", 3, true, false},
		{"only once", "cuda-module-always", "", 2, false, true},
		{"explicit override", "cuda-module-always", "LAZY", 1, false, true},
		{"keep complete PNG", "saved-cuda-module", "", 1, true, true},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			dir := t.TempDir()
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			config := Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: filepath.Join(dir, "outputs")}
			engine, err := NewEngine(config)
			if err != nil {
				t.Fatal(err)
			}
			defer engine.Close()
			log := filepath.Join(dir, "attempts.jsonl")
			t.Setenv("CUDA_MODULE_LOADING", scenario.env)
			t.Setenv("WILDIMAGE_FAKE_CHILD", scenario.mode)
			t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", log)
			result, runErr := engine.Run(t.Context(), testProfile(t, dir), Request{
				Prompt: "literal --prompt ; &", Seed: 42, OutputPath: filepath.Join(config.OutputDir, "image.png"),
			}, nil)
			attempts := imageAttempts(t, log)
			if len(attempts) != scenario.attempts || (result.Path != "") != scenario.saved || (runErr != nil) != scenario.fails {
				t.Fatalf("attempts=%d, output=%+v, error=%v", len(attempts), result, runErr)
			}
			if len(attempts) > 1 && !slices.Equal(attempts[len(attempts)-2], attempts[len(attempts)-1]) {
				t.Fatal("CUDA retry changed the seed, image options or placement")
			}
			if os.Getenv("CUDA_MODULE_LOADING") != scenario.env {
				t.Fatal("CUDA retry changed the parent environment")
			}
		})
	}
}
