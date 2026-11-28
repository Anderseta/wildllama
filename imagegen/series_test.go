package imagegen

import (
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Model a child that dies after rename but before notifying the parent.
func fakeSeriesExit(mode string, args map[string]string) {
	code := uint32(0xc0000005)
	var first, count, group int
	total := 10
	if mode == "series-exit-handoff" {
		total = 1000
	}
	if _, err := fmt.Fscan(os.Stdin, &first, &count, &group); err != nil || count != total-first || (first > 0 && group != 1) {
		os.Exit(8)
	}
	if mode == "series-exit-no-progress" {
		os.Exit(seriesReopenExitCode)
	}
	for i := first; i < first+count; i++ {
		var seed int64
		if _, err := fmt.Fscan(os.Stdin, &seed); err != nil || seed != 57+int64(i) {
			os.Exit(8)
		}
		if mode == "series-exit-twice" && i == 1 {
			os.Exit(int(code))
		}
		file, err := os.Create(filepath.Join(filepath.Dir(args["--output"]), fmt.Sprintf("%d.png", i)))
		if err != nil || png.Encode(file, image.NewGray(image.Rect(0, 0, 64, 64))) != nil || file.Close() != nil {
			os.Exit(8)
		}
		if (mode == "series-exit-once" && i == 0) || (mode == "series-exit-at-end" && i == 9) {
			os.Exit(int(code))
		}
		if mode == "series-exit-handoff" && i+1 < total && (i+1)%250 == 0 {
			if os.Getenv("WILDIMAGE_SERIES_REOPEN") != "1" {
				os.Exit(8)
			}
			os.Exit(seriesReopenExitCode)
		}
	}
}

func TestWindowsSeriesRecovery(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows native exit status")
	}
	for _, tc := range []struct {
		mode                     string
		wantImages, wantAttempts int
	}{
		{"series-exit-once", 10, 2},
		{"series-exit-twice", 1, 2},
		{"series-exit-at-end", 10, 1},
		{"series-exit-handoff", 1000, 4},
		{"series-exit-no-progress", 0, 1},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			dir := t.TempDir()
			p := testProfile(t, dir)
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			cfg := Config{RunnerPath: exe, WorkDir: filepath.Join(dir, "work"), OutputDir: filepath.Join(dir, "outputs"), ModelsDir: filepath.Join(dir, "models")}
			e, err := NewEngine(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			trace := filepath.Join(dir, "attempts.jsonl")
			t.Setenv("WILDIMAGE_FAKE_ATTEMPTS", trace)
			t.Setenv("WILDIMAGE_FAKE_CHILD", tc.mode)
			seen := 0
			series := &ImageSeries{Publish: func(index int, result Result) error {
				if index != seen || result.Seed != int64(57+index) {
					return fmt.Errorf("unexpected index or seed: %d, %d", index, result.Seed)
				}
				seen++
				return nil
			}}
			count := max(10, tc.wantImages)
			for i := 0; i < count; i++ {
				series.Outputs = append(series.Outputs, SeriesOutput{fmt.Sprintf("img%d.png", i), int64(57 + i)})
			}
			request := Request{Prompt: "literal --prompt ; &", OutputPath: filepath.Join(cfg.OutputDir, "img0.png"), Seed: 57, Series: series}
			if tc.mode == "series-exit-handoff" || tc.mode == "series-exit-no-progress" {
				p.VisionEncoder = filepath.Join(dir, "vision.gguf")
				putModel(t, p.VisionEncoder, modelBytes())
				ref := filepath.Join(dir, "reference.png")
				file, err := os.Create(ref)
				if err != nil {
					t.Fatal(err)
				}
				if err = png.Encode(file, image.NewGray(image.Rect(0, 0, 64, 64))); err != nil {
					file.Close()
					t.Fatal(err)
				}
				if err = file.Close(); err != nil {
					t.Fatal(err)
				}
				request.ReferenceImages = []string{ref}
			}
			_, err = e.Run(t.Context(), p, request, nil)
			if (err == nil) != (tc.wantImages == count) || seen != tc.wantImages || len(imageAttempts(t, trace)) != tc.wantAttempts {
				t.Fatalf("%d images, expected %d in %d attempts: %v", seen, tc.wantImages, tc.wantAttempts, err)
			}
		})
	}
}

func TestImageSeriesPublicationAndResume(t *testing.T) {
	dst, staging := t.TempDir(), t.TempDir()
	var published []int64
	plan := &ImageSeries{Outputs: []SeriesOutput{{"one.png", 1}, {"two.png", 8111222333444555666}, {"three.png", 3}},
		Publish: func(i int, r Result) error {
			if i != len(published) || r.Width != 64 || r.Height != 64 || filepath.Dir(r.Path) != filepath.Clean(dst) {
				t.Fatalf("bad result %+v", r)
			}
			published = append(published, r.Seed)
			return nil
		},
	}
	if err := plan.validate(filepath.Join(dst, "one.png"), 1); err != nil {
		t.Fatal(err)
	}
	stopped := false
	run := &imageSeriesRun{plan: plan, staging: staging, destination: dst, width: 64, height: 64, group: 2, stop: func() { stopped = true }, started: time.Now()}
	for _, name := range []string{"0.png", "1.png"} {
		f, err := os.Create(filepath.Join(staging, name))
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(f, image.NewRGBA(image.Rect(0, 0, 64, 64))); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}
	run.notice("WILDIMAGE_GROUP 0 2")
	run.notice("WILDIMAGE_SAVED 0")
	if len(published) != 1 {
		t.Fatal(published)
	}
	input, err := io.ReadAll(run.seedInput())
	if err != nil || string(input) != "1 2 2\n8111222333444555666\n3\n" {
		t.Fatalf("retry protocol %q: %v", input, err)
	}
	run.collect(3) // recover the PNG that was renamed just before the native process exited
	if !reflect.DeepEqual(published, []int64{1, 8111222333444555666}) || stopped {
		t.Fatal(published, run.err)
	}
	if err := os.WriteFile(filepath.Join(staging, "2.png"), []byte("incomplete"), 0600); err != nil {
		t.Fatal(err)
	}
	run.notice("WILDIMAGE_SAVED 2")
	if !stopped || run.err == nil || len(published) != 2 {
		t.Fatalf("invalid partial accepted: %v", run.err)
	}
}

func TestImageSeriesBudgetsAndDestinations(t *testing.T) {
	p := Profile{Width: 1536, Height: 1024}
	if group := seriesGroupSize(p, Request{}); group < 5 || group > 32 {
		t.Fatal(group)
	}
	if seriesGroupSize(p, Request{ReferenceImages: []string{"ref.png"}}) != 1 {
		t.Fatal("seed-dependent references must remain independent")
	}
	p.Width, p.Height, p.MaxRAMMiB = 4096, 4096, 256
	if seriesGroupSize(p, Request{}) != 1 {
		t.Fatal("large canvas ignored memory budget")
	}
	first := filepath.Join(t.TempDir(), "first.png")
	for _, name := range []string{"../x.png", "FIRST.png", "other.txt", "x:y.png"} {
		plan := ImageSeries{Outputs: []SeriesOutput{{"first.png", 3}, {name, 4}}, Publish: func(int, Result) error { return nil }}
		if err := plan.validate(first, 3); err == nil || !strings.Contains(err.Error(), "output") {
			t.Fatal(name, err)
		}
	}
}

func TestQwenStreamGroupBudget(t *testing.T) {
	p := Profile{ModelID: OmniFP8RecipeID, Width: 1536, Height: 1024}
	legacy := seriesGroupSize(p, Request{})
	if legacy != 5 || qwenStreamGroupSize(p, Request{}, legacy) != 12 {
		t.Fatal("unexpected landscape group size")
	}
	if qwenStreamGroupSize(p, Request{ReferenceImages: []string{"ref.png"}}, 1) != 1 {
		t.Fatal("seed-dependent references grouped")
	}
	p.MaxRAMMiB = 256
	if qwenStreamGroupSize(p, Request{}, 1) != 1 {
		t.Fatal("tight RAM budget ignored")
	}
	p.ModelID = "other"
	if qwenStreamGroupSize(p, Request{}, 2) != 2 {
		t.Fatal("unknown model group changed")
	}
	s := imageSeriesRun{plan: &ImageSeries{Outputs: make([]SeriesOutput, 8)}, group: 12, legacyGroup: 5}
	b, err := io.ReadAll(s.seedInput())
	if err != nil || !strings.HasPrefix(string(b), "0 8 5\n") {
		t.Fatal(string(b), err)
	}
	s.group = 2
	b, err = io.ReadAll(s.seedInput())
	if err != nil || !strings.HasPrefix(string(b), "0 8 2\n") {
		t.Fatal(string(b), err)
	}
}
