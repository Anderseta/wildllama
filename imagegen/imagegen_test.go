package imagegen

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	if mode := os.Getenv("WILDIMAGE_FAKE_CHILD"); mode != "" {
		fakeChild(mode)
		os.Exit(0)
	}
	// Test data stays beside this package, even when the invoking machine's
	// default temporary directory is on C:.
	working, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	dir, err := os.MkdirTemp(working, ".wildimage-tests-")
	if err != nil {
		panic(err)
	}
	for _, name := range []string{"TEMP", "TMP", "TMPDIR"} {
		os.Setenv(name, dir)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

func TestProgressNativeCounterPhases(t *testing.T) {
	tests := []struct {
		name, line, phase string
		current, total    int
	}{
		{"tensor count", "loading 42/756 tensors from denoiser.safetensors", "loading", 42, 756},
		{"bytes", "|###       | 42/756 - 512.00B/s\x1b[K", "loading", 42, 756},
		{"megabytes", "|###       | 42/756 - 512.00MB/s\x1b[K", "loading", 42, 756},
		{"gigabytes", "|###       | 42/756 - 1.20GB/s\x1b[K", "loading", 42, 756},
		{"gibibytes", "|###       | 42/756 - 1.20GiB/s\x1b[K", "loading", 42, 756},
		{"mebibytes", "|###       | 42/756 - 512.00MiB/s\x1b[K", "loading", 42, 756},
		{"encoding", "encoding reference image 1/2", "encoding", 1, 2},
		{"decoding", "decoding VAE tile 1/4", "decoding", 1, 4},
		{"saving", "2/2 images saved", "saving", 2, 2},
		{"seconds per iteration", "|====>     | 1/25 - 0.20s/it\x1b[K", "sampling", 1, 25},
		{"iterations per second", "|====>     | 1/25 - 5.00it/s\x1b[K", "sampling", 1, 25},
		{"unlabelled count", "1/25", "sampling", 1, 25},
		{"conditioning cache hit", "conditioning disk cache hit", "conditioning_cache_hit", 0, 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var updates []Progress
			log := progressLog{start: time.Now(), callback: func(update Progress) { updates = append(updates, update) }}
			if _, err := log.Write([]byte(test.line + "\r")); err != nil {
				t.Fatal(err)
			}
			if len(updates) != 1 || updates[0].Phase != test.phase || updates[0].Current != test.current || updates[0].Total != test.total {
				t.Fatalf("native progress %q: %#v", test.line, updates)
			}
		})
	}
}
func fakeChild(mode string) {
	if path := os.Getenv("WILDIMAGE_FAKE_ARGS"); path != "" {
		encoded, _ := json.Marshal(os.Args[1:])
		if err := os.WriteFile(path, encoded, 0600); err != nil {
			os.Exit(8)
		}
	}
	values := map[string]string{}
	for i := 1; i+1 < len(os.Args); i++ {
		if strings.HasPrefix(os.Args[i], "--") {
			values[os.Args[i]] = os.Args[i+1]
		}
	}
	time.Sleep(25 * time.Millisecond)
	fmt.Fprintln(os.Stderr, "loading model components")
	if mode == "wait" {
		time.Sleep(30 * time.Second)
		return
	}
	prompt, err := os.ReadFile(values["--prompt-file"])
	if err != nil || !strings.Contains(string(prompt), "literal --prompt ; &") {
		os.Exit(3)
	}
	for step := 1; step <= 8; step++ {
		fmt.Fprintf(os.Stderr, "%d/8 - 12.5 it/s\r", step)
	}
	w, _ := strconv.Atoi(values["--width"])
	h, _ := strconv.Atoi(values["--height"])
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	im.Set(0, 0, color.RGBA{R: 255, A: 255})
	f, err := os.Create(values["--output"])
	if err != nil {
		os.Exit(4)
	}
	if err := png.Encode(f, im); err != nil {
		os.Exit(5)
	}
	f.Close()
	if mode == "truncated-output" {
		_ = os.Truncate(values["--output"], 33)
	}
	fmt.Fprintln(os.Stderr, "1/1 images saved")
	if mode == "late-failure" {
		os.Exit(7)
	}
	if mode == "saved-wait" {
		time.Sleep(30 * time.Second)
	}
}
func modelBytes() []byte {
	data := make([]byte, 4096)
	copy(data, "GGUF")
	for i := 8; i < len(data); i++ {
		data[i] = byte(i * 17)
	}
	return data
}
func putModel(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}
func digest(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
func testProfile(t *testing.T, dir string) Profile {
	t.Helper()
	p := DefaultProfile(DefaultRecipeID)
	p.Width = 64
	p.Height = 64
	p.DiffusionModel = filepath.Join(dir, "diffusion.gguf")
	p.TextEncoder = filepath.Join(dir, "encoder.gguf")
	p.VAE = filepath.Join(dir, "vae.safetensors")
	putModel(t, p.DiffusionModel, modelBytes())
	putModel(t, p.TextEncoder, modelBytes())
	vae := make([]byte, 16)
	binary.LittleEndian.PutUint64(vae, 2)
	copy(vae[8:], "{}")
	putModel(t, p.VAE, vae)
	return p
}
func TestRecipesAndArguments(t *testing.T) {
	list := Recipes()
	if len(list) != 2 || list[0].ID != DefaultRecipeID {
		t.Fatal("Q5 default/Q4 fallback missing")
	}
	for _, r := range list {
		roles := map[string]bool{}
		for _, c := range r.Components {
			roles[c.Role] = true
			if len(c.SHA256) != 64 || len(c.Revision) != 40 || c.Size <= 0 {
				t.Fatal("unpinned component")
			}
		}
		if !roles["vae"] || !roles["text_encoder"] || !roles["diffusion_model"] || !roles["vision_encoder"] {
			t.Fatal("incomplete image recipe")
		}
	}
	p := DefaultProfile(DefaultRecipeID)
	p.DiffusionModel = "diff.gguf"
	p.TextEncoder = "encoder.gguf"
	p.VAE = "vae.safetensors"
	if err := ValidateProfile(p); err != nil {
		t.Fatal(err)
	}
	args := strings.Join(buildArguments(p, Request{}, "job", "out.png", 42), "|")
	for _, required := range []string{"--sigmas|" + TurboSigmas, "--steps|8", "--cfg-scale|1", "--sampling-method|euler", "--max-vram|12.0000", "--backend|cuda0", "--prompt-file|"} {
		if !strings.Contains(args, required) {
			t.Fatalf("missing %s", required)
		}
	}
	if strings.Contains(args, "--cache") || strings.Contains(args, "--llm_vision") {
		t.Fatal("unrequested cache/edit args")
	}
	p.VisionEncoder = "same-encoder-mmproj.gguf"
	editArgs := strings.Join(buildArguments(p, Request{ReferenceImages: []string{"first.png", "second.png"}}, "job", "out.png", 42), "|")
	if !strings.Contains(editArgs, "--llm_vision|same-encoder-mmproj.gguf") || !strings.Contains(editArgs, "-r|first.png|-r|second.png") {
		t.Fatal("reference edit arguments changed", editArgs)
	}
	p.Width = 1000
	if ValidateProfile(p) == nil {
		t.Fatal("accepted invalid dimension")
	}
	p.Width = 1024
	p.Steps = 7
	if ValidateProfile(p) == nil {
		t.Fatal("accepted inconsistent Turbo steps/sigmas")
	}
}
func TestDownloadResumeAndIgnoredRange(t *testing.T) {
	for _, ignore := range []bool{false, true} {
		t.Run(fmt.Sprint(ignore), func(t *testing.T) {
			data := modelBytes()
			dir := t.TempDir()
			target := filepath.Join(dir, "weight.gguf")
			putModel(t, target+".part", data[:123])
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("Range") != "bytes=123-" {
					t.Errorf("resume range %q", r.Header.Get("Range"))
				}
				if !ignore {
					w.Header().Set("Content-Range", fmt.Sprintf("bytes 123-%d/%d", len(data)-1, len(data)))
					w.WriteHeader(http.StatusPartialContent)
					w.Write(data[123:])
				} else {
					w.Write(data)
				}
			}))
			defer server.Close()
			c := InstalledComponent{Component: Component{Repo: "owner/repo", Revision: strings.Repeat("a", 40), Filename: "weight.gguf", SHA256: digest(data), Size: int64(len(data))}, Path: target}
			if err := downloadComponent(context.Background(), server.Client(), server.URL, c, nil); err != nil {
				t.Fatal(err)
			}
			actual, _ := os.ReadFile(target)
			if string(actual) != string(data) {
				t.Fatal("resume changed bytes")
			}
			if _, err := os.Stat(target + ".part"); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("partial survived verified publication")
			}
		})
	}
}
func TestDownloadRejectsRangeAndChecksum(t *testing.T) {
	data := modelBytes()
	for _, badRange := range []bool{true, false} {
		t.Run(fmt.Sprint(badRange), func(t *testing.T) {
			dir := t.TempDir()
			target := filepath.Join(dir, "w.gguf")
			if badRange {
				putModel(t, target+".part", data[:123])
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if badRange {
					w.Header().Set("Content-Range", fmt.Sprintf("bytes 0-%d/%d", len(data)-1, len(data)))
					w.WriteHeader(206)
				}
				w.Write(data)
			}))
			defer server.Close()
			c := InstalledComponent{Component: Component{Repo: "o/r", Revision: "pin", Filename: "w.gguf", SHA256: strings.Repeat("0", 64), Size: int64(len(data))}, Path: target}
			if downloadComponent(context.Background(), server.Client(), server.URL, c, nil) == nil {
				t.Fatal("accepted invalid response/hash")
			}
			if _, err := os.Stat(target); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("invalid payload was published")
			}
		})
	}
}
func TestDownloadCancelKeepsPartial(t *testing.T) {
	data := modelBytes()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(data[:16])
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()
	target := filepath.Join(t.TempDir(), "w.gguf")
	c := InstalledComponent{Component: Component{Repo: "o/r", Revision: "pin", Filename: "w.gguf", SHA256: digest(data), Size: int64(len(data))}, Path: target}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if downloadComponent(ctx, server.Client(), server.URL, c, nil) == nil {
		t.Fatal("cancellation ignored")
	}
	if info, err := os.Stat(target + ".part"); err != nil || info.Size() != 16 {
		t.Fatal("resumable bytes lost", err)
	}
	if _, err := os.Stat(target); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("canceled download published")
	}
}
func TestImportAndReadOnlyStatus(t *testing.T) {
	dir := t.TempDir()
	p := testProfile(t, dir)
	b := Bundle{ID: "local-image", Name: "Local", Profile: p}
	for role, path := range map[string]string{"diffusion_model": p.DiffusionModel, "text_encoder": p.TextEncoder, "vae": p.VAE} {
		data, _ := os.ReadFile(path)
		b.Components = append(b.Components, InstalledComponent{Component: Component{Role: role, Size: int64(len(data)), SHA256: digest(data), Required: true}, Path: path})
	}
	manifest := filepath.Join(dir, ManifestName)
	blob, _ := json.Marshal(b)
	os.WriteFile(manifest, blob, 0600)
	models := filepath.Join(dir, "models")
	installed, err := ImportLocal(context.Background(), models, manifest)
	if err != nil || !installed.Ready {
		t.Fatal("import failed", err)
	}
	if installed.Profile.DiffusionModel != p.DiffusionModel {
		t.Fatal("import duplicated/moved weights")
	}
	catalog, _, err := Catalog(models)
	if err != nil || len(catalog) != 1 || !catalog[0].Ready {
		t.Fatal("catalog", err)
	}
	config := Config{RunnerPath: filepath.Join(dir, "missing-runner.exe"), WorkDir: filepath.Join(dir, "never-created-work"), ModelsDir: models, OutputDir: filepath.Join(dir, "never-created-output")}
	engine, err := NewEngine(config)
	if err != nil {
		t.Fatal(err)
	}
	if engine.Status().Ready {
		t.Fatal("missing runner marked ready")
	}
	if _, err := os.Stat(config.WorkDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("status created a work directory")
	}
}

func TestComponentPartialStatusUsesExistingFileSizes(t *testing.T) {
	dir := t.TempDir()
	p := testProfile(t, dir)
	b := Bundle{ID: "partial-image", Profile: p, Components: []InstalledComponent{
		{Component: Component{Role: "diffusion_model", Size: 4096, Required: true}, Path: p.DiffusionModel},
		{Component: Component{Role: "text_encoder", Size: 4096, Required: true}, Path: p.TextEncoder},
		{Component: Component{Role: "vae", Size: 16, Required: true}, Path: p.VAE},
		{Component: Component{Role: "vision_encoder", Size: 4096}, Path: filepath.Join(dir, "vision.gguf")},
	}}
	if err := os.Remove(p.TextEncoder); err != nil {
		t.Fatal(err)
	}
	putModel(t, p.TextEncoder+".part", modelBytes()[:123])
	refreshBundle(&b)
	if b.Ready || !b.Components[0].Ready || b.Components[0].DownloadedBytes != 4096 || b.Components[0].DownloadState != "ready" {
		t.Fatal("completed file status", b)
	}
	if b.Components[1].Ready || b.Components[1].DownloadedBytes != 123 || b.Components[1].DownloadState != "partial" {
		t.Fatal("partial file status", b.Components[1])
	}
	if b.Components[3].Ready || b.Components[3].DownloadedBytes != 0 || b.Components[3].DownloadState != "missing" || b.Profile.VisionEncoder != "" {
		t.Fatal("optional projector status", b.Components[3])
	}
}
func TestIsolatedProcessSuccessAndCancellation(t *testing.T) {
	for _, mode := range []string{"generate", "wait"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			p := testProfile(t, dir)
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
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			request := Request{Prompt: "literal --prompt ; & $(do not execute)", Seed: 42, OutputPath: filepath.Join(dir, "outputs", "result.png")}
			var phases []string
			result, err := engine.Run(ctx, p, request, func(v Progress) {
				phases = append(phases, v.Phase)
				if mode == "wait" && v.Phase == "loading" {
					cancel()
				}
			})
			if mode == "wait" {
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancel: %v", err)
				}
				if engine.Status().Running {
					t.Fatal("canceled process remains running")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if result.Width != 64 || result.Height != 64 || result.Seed != 42 || result.MIMEType != "image/png" {
				t.Fatal("output metadata", result)
			}
			if !strings.Contains(strings.Join(phases, ","), "sampling") {
				t.Fatal("no detailed progress", phases)
			}
			for _, name := range []string{"TEMP", "TMP", "TMPDIR", "CUDA_CACHE_PATH", "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "XDG_CACHE_HOME"} {
				env := isolatedEnvironment(engine.config, engine.config.WorkDir)
				found := false
				for _, entry := range env {
					if strings.HasPrefix(entry, name+"=") {
						found = true
						if !within(filepath.Clean(dir), filepath.Clean(strings.TrimPrefix(entry, name+"="))) {
							t.Fatal("cache escaped test root", entry)
						}
					}
				}
				if !found {
					t.Fatal("missing child cache", name)
				}
			}
		})
	}
}
