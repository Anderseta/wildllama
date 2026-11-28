package imagegen

import (
	"errors"
	"fmt"
	"image"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const MaxImageSeries = 1000

type SeriesOutput struct {
	Filename string
	Seed     int64
}

// Publish runs in output order, only for completely decoded and committed files.
type ImageSeries struct {
	Outputs []SeriesOutput
	Publish func(int, Result) error
}

func (s *ImageSeries) validate(first string, seed int64) error {
	if len(s.Outputs) < 2 || len(s.Outputs) > MaxImageSeries || s.Publish == nil ||
		s.Outputs[0].Seed != seed || s.Outputs[0].Filename != filepath.Base(first) {
		return errors.New("invalid image series")
	}
	names := make(map[string]bool, len(s.Outputs))
	for _, item := range s.Outputs {
		name := strings.ToLower(item.Filename)
		if item.Seed < 0 || !filepath.IsLocal(name) || filepath.Base(name) != name ||
			strings.ContainsAny(name, "/\\:%") || filepath.Ext(name) != ".png" || names[name] {
			return errors.New("invalid or duplicate image series output")
		}
		names[name] = true
		path := filepath.Join(filepath.Dir(first), item.Filename)
		if _, err := os.Lstat(path); err == nil {
			return errors.New("image series output already exists")
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

// A group holds final latents and decoded pixels on the host, but evaluates
// only one denoising trajectory on the device at a time. Reference pipelines
// may sample the reference latent with the individual image seed.
func seriesGroupSize(p Profile, r Request) int {
	if len(r.ReferenceImages) != 0 || r.Inpaint != nil {
		return 1
	}
	budget := seriesRasterBudget(p)
	return int(max(1, min(32, budget/(int64(p.Width)*int64(p.Height)*24))))
}

func seriesRasterBudget(p Profile) int64 {
	budget := int64(192 << 20)
	if p.MaxRAMMiB > 0 {
		budget = min(budget, (int64(p.MaxRAMMiB)<<20)/16)
	}
	return budget
}

// Streamed groups hold one active decoder and PNG writer. Only the final
// latents accumulate. Limit the time until the first gallery image as well.
func qwenStreamGroupSize(p Profile, r Request, legacy int) int {
	if p.ModelID != OmniFP8RecipeID || len(r.ReferenceImages) != 0 || r.Inpaint != nil {
		return legacy
	}
	pixels := int64(p.Width) * int64(p.Height)
	room := seriesRasterBudget(p) - 48*pixels
	return max(legacy, int(min(12, room/(2*pixels))))
}

type imageSeriesRun struct {
	plan                                                        *ImageSeries
	staging, destination, revision                              string
	width, height, group, legacyGroup, first, activeCount, next int
	started                                                     time.Time
	stop                                                        func()
	compose                                                     func(image.Image, string) (image.Point, error)
	last                                                        Result
	err                                                         error
}

func (s *imageSeriesRun) seedInput() io.Reader {
	var input strings.Builder
	group := s.group
	if s.legacyGroup > 0 {
		group = min(group, s.legacyGroup)
	}
	fmt.Fprintf(&input, "%d %d %d\n", s.next, len(s.plan.Outputs)-s.next, group)
	for _, item := range s.plan.Outputs[s.next:] {
		fmt.Fprintln(&input, item.Seed)
	}
	return strings.NewReader(input.String())
}

func (s *imageSeriesRun) notice(line string) bool {
	if rest, ok := strings.CutPrefix(line, "WILDIMAGE_GROUP "); ok {
		var first, count int
		if _, err := fmt.Sscan(rest, &first, &count); err == nil && first == s.next && count > 0 && count <= s.group && count <= len(s.plan.Outputs)-first {
			s.first, s.activeCount = first, count
			s.started = time.Now()
		}
		return true
	}
	if rest, ok := strings.CutPrefix(line, "WILDIMAGE_SAVED "); ok {
		if i, err := strconv.Atoi(rest); err == nil && i >= s.next && i < len(s.plan.Outputs) {
			s.collect(i + 1)
		}
		return true
	}
	return false
}

func (s *imageSeriesRun) collect(limit int) {
	for s.next < limit && s.err == nil {
		path := filepath.Join(s.staging, strconv.Itoa(s.next)+".png")
		if _, err := os.Lstat(path); errors.Is(err, os.ErrNotExist) {
			return
		}
		s.err = s.publish(path)
		if s.err != nil {
			s.stop()
			return
		}
		s.next++
	}
}

func (s *imageSeriesRun) publish(path string) (err error) {
	// The subprocess pipe invokes this on its own goroutine.
	defer func() {
		if cause := recover(); cause != nil {
			err = fmt.Errorf("image series publication failed: %v", cause)
		}
	}()
	img, err := decodeOutputPNG(path, s.width, s.height)
	if err != nil {
		return err
	}
	item := s.plan.Outputs[s.next]
	dest := filepath.Join(s.destination, item.Filename)
	if _, err := os.Lstat(dest); err == nil {
		return errors.New("image series destination already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	size := image.Pt(s.width, s.height)
	if s.compose != nil {
		size, err = s.compose(img, dest)
		if err == nil {
			_ = os.Remove(path)
		}
	} else {
		err = os.Rename(path, dest)
	}
	if err != nil {
		return err
	}
	s.last = Result{Path: dest, Width: size.X, Height: size.Y, Seed: item.Seed, MIMEType: "image/png", RunnerRevision: s.revision,
		DurationSeconds: time.Since(s.started).Seconds() / float64(max(1, s.activeCount))}
	return s.plan.Publish(s.next, s.last)
}
