package imagegen

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestImageRAMBudgetDefaultsAndValidation(t *testing.T) {
	profile := DefaultProfile(DefaultRecipeID)
	if profile.MaxRAMMiB != 0 || profile.MaxVRAMMiB != 0 || profile.VRAMReserveMiB != 256 || !profile.OffloadToCPU || !profile.StageComponents {
		t.Fatal("default image policy must allow bounded automatic RAM offload", profile)
	}
	for _, limit := range []int{0, 1, 15360, 32768, 1048576} {
		candidate := profile
		candidate.MaxRAMMiB = limit
		if err := ValidateProfile(candidate); err != nil {
			t.Fatalf("valid RAM budget %d: %v", limit, err)
		}
		data, err := json.Marshal(candidate)
		if err != nil {
			t.Fatal(err)
		}
		var loaded Profile
		if err := json.Unmarshal(data, &loaded); err != nil || loaded.MaxRAMMiB != limit {
			t.Fatalf("RAM budget did not survive JSON: %d, %v, %#v", limit, err, loaded)
		}
		if loaded.MaxVRAMMiB != profile.MaxVRAMMiB || loaded.Steps != profile.Steps || loaded.Sigmas != profile.Sigmas {
			t.Fatal("RAM budget changed sampling or the GPU budget", loaded)
		}
	}
	for _, limit := range []int{-1, 1048577} {
		candidate := profile
		candidate.MaxRAMMiB = limit
		if err := ValidateProfile(candidate); err == nil {
			t.Fatalf("accepted invalid RAM budget %d", limit)
		}
		candidate = profile
		candidate.VRAMReserveMiB = limit
		if err := ValidateProfile(candidate); err == nil {
			t.Fatalf("accepted invalid free VRAM margin %d", limit)
		}
	}
}

func TestImageMemoryArgumentsPreserveExactMiBLimits(t *testing.T) {
	for _, limit := range []int{0, 1, 1025, 15360, 1048576} {
		profile := DefaultProfile(DefaultRecipeID)
		profile.MaxRAMMiB, profile.MaxVRAMMiB, profile.VRAMReserveMiB = limit, limit, limit
		args := buildArguments(profile, Request{}, "fixture", "output.png", 42)
		for _, flag := range []string{"--max-ram", "--max-vram", "--vram-reserve"} {
			index := slices.Index(args, flag)
			if limit == 0 && flag == "--max-vram" {
				if index >= 0 {
					t.Fatal("automatic GPU budget must retain native available-memory behavior")
				}
				continue
			}
			if index < 0 || index+1 >= len(args) {
				t.Fatal("missing explicit memory budget", flag, args)
			}
			gib, err := strconv.ParseFloat(args[index+1], 64)
			if err != nil || gib*(1<<30) != float64(int64(limit)<<20) {
				t.Fatalf("%s rounded %d MiB to %q: %v", flag, limit, args[index+1], err)
			}
		}
	}
}

func imageAttempts(t *testing.T, path string) [][]string {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var attempts [][]string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var args []string
		if err := json.Unmarshal(scanner.Bytes(), &args); err != nil {
			t.Fatal(err)
		}
		attempts = append(attempts, args)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return attempts
}

func imageArgument(t *testing.T, args []string, flag string) string {
	t.Helper()
	index := slices.Index(args, flag)
	if index < 0 || index+1 >= len(args) {
		t.Fatal("missing child argument", flag, args)
	}
	return args[index+1]
}

func imageInvariantArguments(args []string) []string {
	var result []string
	for index := 0; index < len(args); index++ {
		switch args[index] {
		case "--backend", "--params-backend":
			index++
		case "--offload-to-cpu", "--disable-segmented-compute":
		default:
			result = append(result, args[index])
		}
	}
	return result
}

func TestRAMWeightOffloadKeepsGPUComputeWhenItFits(t *testing.T) {
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
	path := filepath.Join(dir, "attempts.jsonl")
	t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", path)
	t.Setenv("WILDIMAGE_FAKE_CHILD", "gpu-oom-then-ram")
	var phases []string
	_, err = engine.Run(t.Context(), profile, Request{Prompt: "literal --prompt ; &", Seed: -1, OutputPath: filepath.Join(dir, "outputs", "image.png")}, func(progress Progress) { phases = append(phases, progress.Phase) })
	if err != nil {
		t.Fatal(err)
	}
	attempts := imageAttempts(t, path)
	if len(attempts) != 2 || imageArgument(t, attempts[1], "--backend") != "cuda0" || imageArgument(t, attempts[1], "--params-backend") != "cpu" || slices.Contains(phases, "cpu_fallback") {
		t.Fatal("fitting RAM/GPU offload unnecessarily switched compute to CPU", attempts, phases)
	}
	if !slices.Equal(imageInvariantArguments(attempts[0]), imageInvariantArguments(attempts[1])) {
		t.Fatal("RAM offload changed image parameters", attempts)
	}
}

func TestGPUResidentPlacementPreservesExplicitCPUComponents(t *testing.T) {
	for _, pair := range []struct{ encoder, vae bool }{{true, false}, {false, true}, {true, true}} {
		profile := DefaultProfile(DefaultRecipeID)
		profile.EncoderOnCPU, profile.VAEOnCPU = pair.encoder, pair.vae
		args := buildArguments(profile, Request{}, "job", "output.png", 42)
		compute := imageArgument(t, args, "--backend")
		params := imageArgument(t, args, "--params-backend")
		if compute != params || strings.Contains(params, "te=cpu") != pair.encoder || strings.Contains(params, "vae=cpu") != pair.vae {
			t.Fatal("GPU-first placement changed a selected CPU component", pair, args)
		}
	}
}

func TestClassifiedGPUFailureRetriesOnCPUWithSameSeedReferencesAndQuality(t *testing.T) {
	for _, count := range []int{1, 2, 4} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			dir := t.TempDir()
			profile := testProfile(t, dir)
			profile.VisionEncoder = profile.DiffusionModel
			profile.MaxRAMMiB, profile.MaxVRAMMiB = 1025, 1025
			before := profile
			refs := make([]string, count)
			for i := range refs {
				refs[i] = filepath.Join(dir, fmt.Sprintf("reference-%d.png", count-i))
				referenceSizeHeader(t, refs[i], 64, 64)
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
			attemptPath := filepath.Join(dir, "attempts.jsonl")
			t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", attemptPath)
			t.Setenv("WILDIMAGE_FAKE_CHILD", "gpu-oom-then-cpu")
			request := Request{Prompt: "literal --prompt ; & unchanged", NegativePrompt: "unchanged negative", Seed: -1,
				ReferenceImages: refs, OutputPath: filepath.Join(dir, "outputs", "image.png")}
			var phases []string
			result, err := engine.Run(t.Context(), profile, request, func(progress Progress) { phases = append(phases, progress.Phase) })
			if err != nil || result.Path != request.OutputPath || result.Seed < 0 {
				t.Fatalf("CPU recovery failed: %#v, %v", result, err)
			}
			attempts := imageAttempts(t, attemptPath)
			if len(attempts) != 3 || imageArgument(t, attempts[0], "--backend") != "cuda0" || imageArgument(t, attempts[1], "--backend") != "cuda0" || imageArgument(t, attempts[2], "--backend") != "cpu" {
				t.Fatal("expected GPU-resident, RAM/GPU, then CPU", attempts)
			}
			for i, args := range attempts {
				if !slices.Equal(imageInvariantArguments(attempts[0]), imageInvariantArguments(args)) {
					t.Fatal("recovery altered precision, sampling, paths or other parameters", attempts)
				}
				if (i == 0) != slices.Contains(args, "--disable-segmented-compute") {
					t.Fatal("GPU-only segmentation policy leaked into fallback", args)
				}
			}
			if imageArgument(t, attempts[2], "--seed") != strconv.FormatInt(result.Seed, 10) || !reflect.DeepEqual(before, profile) {
				t.Fatal("CPU recovery redrew the random seed or changed the saved profile", attempts, profile)
			}
			var actualRefs []string
			for i, value := range attempts[2] {
				if value == "-r" {
					actualRefs = append(actualRefs, attempts[2][i+1])
				}
			}
			if !slices.Equal(actualRefs, refs) || !slices.Contains(phases, "ram_offload") || !slices.Contains(phases, "cpu_fallback") {
				t.Fatal("ordered references or recovery progress were lost", actualRefs, phases)
			}
			if engine.Status().Running {
				t.Fatal("CPU recovery left a child process active")
			}
		})
	}
}

func TestGPUFallbackNeverRepeatsRAMFailureBadModelDisabledOffloadOrSavedPNG(t *testing.T) {
	for _, scenario := range []struct {
		mode     string
		offload  bool
		attempts int
		saved    bool
	}{
		{"ram-budget", true, 1, false},
		{"invalid-model", true, 1, false},
		{"gpu-oom-then-cpu", false, 1, false},
		{"gpu-oom-always", true, 3, false},
		{"gpu-oom-then-ram-budget", true, 2, false},
		{"saved-gpu-oom", true, 1, true},
	} {
		t.Run(scenario.mode+fmt.Sprint(scenario.offload), func(t *testing.T) {
			dir := t.TempDir()
			profile := testProfile(t, dir)
			profile.OffloadToCPU = scenario.offload
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			engine, err := NewEngine(Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: filepath.Join(dir, "outputs")})
			if err != nil {
				t.Fatal(err)
			}
			defer engine.Close()
			path := filepath.Join(dir, "attempts.jsonl")
			t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", path)
			t.Setenv("WILDIMAGE_FAKE_CHILD", scenario.mode)
			result, runErr := engine.Run(t.Context(), profile, Request{Prompt: "literal --prompt ; &", Seed: 42, OutputPath: filepath.Join(dir, "outputs", "image.png")}, nil)
			if runErr == nil || len(imageAttempts(t, path)) != scenario.attempts || (result.Path != "") != scenario.saved {
				t.Fatalf("invalid recovery escalation or lost original: %#v, %v", result, runErr)
			}
			var exit *exec.ExitError
			if !errors.As(runErr, &exit) {
				t.Fatal("native failure classification was lost", runErr)
			}
			if scenario.mode == "ram-budget" && exit.ExitCode() != RAMBudgetExitCode {
				t.Fatal("RAM budget error became another failure", runErr)
			}
		})
	}
}

func TestGPUFallbackDoesNotRestartAfterCancellation(t *testing.T) {
	for _, cancelPhase := range []string{"loading", "ram_offload", "cpu_fallback"} {
		t.Run(cancelPhase, func(t *testing.T) {
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
			path := filepath.Join(dir, "attempts.jsonl")
			t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", path)
			t.Setenv("WILDIMAGE_FAKE_CHILD", "gpu-oom-then-cpu")
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			_, err = engine.Run(ctx, profile, Request{Prompt: "literal --prompt ; &", Seed: 42, OutputPath: filepath.Join(dir, "outputs", "image.png")}, func(progress Progress) {
				if progress.Phase == cancelPhase {
					cancel()
				}
			})
			wantAttempts := 1
			if cancelPhase == "cpu_fallback" {
				wantAttempts = 2
			}
			if !errors.Is(err, context.Canceled) || len(imageAttempts(t, path)) != wantAttempts || engine.Status().Running {
				t.Fatal("cancellation started a second child or left one active", err)
			}
		})
	}
}

func TestCPUAdmissionStartsOneCPUProcessWithoutAGPUAttempt(t *testing.T) {
	dir := t.TempDir()
	profile := testProfile(t, dir)
	profile.MaxRAMMiB, profile.MaxVRAMMiB = 15360, 1025
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	engine, err := NewEngine(Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: filepath.Join(dir, "outputs")})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	path := filepath.Join(dir, "attempts.jsonl")
	t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", path)
	t.Setenv("WILDIMAGE_FAKE_CHILD", "gpu-oom-then-cpu")
	var phases []string
	request := Request{Prompt: "literal --prompt ; &", Seed: 42, CPUOnly: true, OutputPath: filepath.Join(dir, "outputs", "image.png")}
	result, err := engine.Run(t.Context(), profile, request, func(progress Progress) { phases = append(phases, progress.Phase) })
	if err != nil || result.Path != request.OutputPath {
		t.Fatal("initial CPU admission failed", result, err)
	}
	attempts := imageAttempts(t, path)
	if len(attempts) != 1 || imageArgument(t, attempts[0], "--backend") != "cpu" || slices.Contains(phases, "cpu_fallback") {
		t.Fatal("CPU admission incorrectly attempted a GPU generation", attempts, phases)
	}
	if imageArgument(t, attempts[0], "--max-ram") != "15" || imageArgument(t, attempts[0], "--max-vram") != "1.0009765625" {
		t.Fatal("initial CPU admission changed requested memory caps", attempts)
	}
	encoded, err := json.Marshal(request)
	if err != nil || strings.Contains(string(encoded), "CPUOnly") || strings.Contains(string(encoded), "cpu_only") {
		t.Fatal("internal admission leaked into the public image request", string(encoded), err)
	}
}

func TestImageRuntimeWithoutRAMBudgetIsRejectedBeforeExecution(t *testing.T) {
	dir := t.TempDir()
	profile := testProfile(t, dir)
	runner := filepath.Join(dir, "old-runner.exe")
	putModel(t, runner, []byte("old-runtime"))
	engine, err := NewEngine(Config{RunnerPath: runner, WorkDir: filepath.Join(dir, "work"), ModelsDir: filepath.Join(dir, "models"), OutputDir: filepath.Join(dir, "outputs")})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	if status := engine.Status(); status.Ready || status.RAMBudgetSupported || !strings.Contains(status.Error, "bounded RAM") {
		t.Fatal("old runtime advertised a managed RAM limit", status)
	}
	_, err = engine.Run(t.Context(), profile, Request{Prompt: "literal --prompt ; &", OutputPath: filepath.Join(dir, "outputs", "image.png")}, nil)
	if err == nil || !strings.Contains(err.Error(), "bounded RAM") {
		t.Fatal("old runtime started without a RAM guard", err)
	}
	if _, err := os.Stat(engine.config.WorkDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("unsupported runtime created a job directory", err)
	}
}

func TestLegacyImageProfileUsesBoundedAutoWithoutChangingPlacement(t *testing.T) {
	profile := DefaultProfile(DefaultRecipeID)
	profile.OffloadToCPU = false
	profile.MaxVRAMMiB = 8192
	data, err := json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	var legacy map[string]json.RawMessage
	if err := json.Unmarshal(data, &legacy); err != nil {
		t.Fatal(err)
	}
	delete(legacy, "max_ram_mib")
	delete(legacy, "vram_reserve_mib")
	data, err = json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	var loaded Profile
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatal(err)
	}
	if loaded.MaxRAMMiB != 0 || loaded.OffloadToCPU || !loaded.StageComponents || loaded.MaxVRAMMiB != 8192 || loaded.VRAMReserveMiB != 256 {
		t.Fatal("legacy profile placement or budget was changed", loaded)
	}
	if err := ValidateProfile(loaded); err != nil {
		t.Fatal(err)
	}
}
