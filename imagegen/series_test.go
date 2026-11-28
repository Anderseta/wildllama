package imagegen

import (
	"image"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

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
