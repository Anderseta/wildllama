package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/imagegen"
)

type videoDetails struct {
	Prompt              string           `json:"prompt"`
	Negative            string           `json:"negative_prompt"`
	Model               string           `json:"model"`
	Width               int              `json:"width"`
	Height              int              `json:"height"`
	Steps               int              `json:"steps"`
	CFG                 float64          `json:"cfg_scale"`
	Sampler             string           `json:"sampler"`
	Scheduler           string           `json:"scheduler"`
	Backend             string           `json:"backend,omitempty"`
	AttentionPreference string           `json:"attention_preference,omitempty"`
	Timings             *imageJobTimings `json:"timings,omitempty"`
}
type storedVideo struct {
	videoRecord
	Details *videoDetails `json:"details,omitempty"`
}

func (r *storedVideo) fromJob(j *imageJob) {
	if j.Generation == nil {
		return
	}
	g := j.Generation
	p := g.Profile
	r.Details = &videoDetails{Prompt: g.Prompt, Negative: g.NegativePrompt, Model: filepath.Base(p.DiffusionModel), Width: p.Width, Height: p.Height, Steps: p.Steps, CFG: p.CFGScale, Sampler: p.Sampler, Scheduler: p.Scheduler, Backend: j.ComputeBackend, Timings: j.imageTimingSnapshot(time.Now())}
	if j.Video != nil {
		r.Details.AttentionPreference = j.Video.AttentionPreference()
	}
	if r.GenerationSeconds <= 0 && r.Details.Timings != nil {
		r.GenerationSeconds = r.Details.Timings.RunSeconds
	}
}

func (v *imageService) videoRecordPath(id string) string {
	return filepath.Join(v.root, "video-records", id+".json")
}

func (v *imageService) readVideo(id string) (storedVideo, error) {
	return v.readVideoFile(id, v.videoRecordPath(id))
}

func (v *imageService) readVideoFile(id, path string) (storedVideo, error) {
	var r storedVideo
	if !imageSafeID(id) {
		return r, fs.ErrNotExist
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return r, err
	}
	if err = json.Unmarshal(b, &r); err != nil {
		return r, err
	}
	if r.ID != id || !filepath.IsAbs(r.Path) || filepath.Base(r.Path) != "video.webm" || filepath.Base(filepath.Dir(r.Path)) != id {
		return r, errors.New("registro de vídeo inválido")
	}
	return r, nil
}

// Old records have the prompt only in the bounded job history. Preserve it
// before jobs can be pruned, and keep the frequent queue response small.
func (v *imageService) preserveVideoDetails() {
	for _, j := range v.jobs {
		if j.VideoFile == nil || j.Generation == nil || !imageSafeID(j.ID) {
			continue
		}
		r, err := v.readVideo(j.ID)
		if errors.Is(err, fs.ErrNotExist) {
			j.VideoFile = nil
			j.HistoryCleared = true
			continue
		}
		if err != nil {
			j.PersistenceError = err.Error()
			continue
		}
		if r.Details == nil {
			r.fromJob(j)
			if err := writeImageJSON(v.videoRecordPath(j.ID), r); err != nil {
				j.PersistenceError = err.Error()
			}
		}
	}
}

func (v *imageService) loadVideoGalleryLocked() error {
	if v.videoArchive != nil {
		return nil
	}
	entries, err := os.ReadDir(filepath.Join(v.root, "video-records"))
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	archive := make(map[string]videoRecord, len(entries))
	for _, e := range entries {
		if id, pending := strings.CutSuffix(e.Name(), ".removing"); pending && imageSafeID(id) && !e.IsDir() {
			path := filepath.Join(v.root, "video-records", e.Name())
			record, readErr := v.readVideoFile(id, path)
			if readErr == nil {
				readErr = removeVideoOutput(record.videoRecord)
			}
			if readErr == nil {
				readErr = os.Remove(path)
			}
			if readErr != nil {
				return fmt.Errorf("finalizar exclusão do vídeo: %w", readErr)
			}
			continue
		}
		id, ok := strings.CutSuffix(e.Name(), ".json")
		if !ok || e.IsDir() || !imageSafeID(id) {
			continue
		}
		r, err := v.readVideo(id)
		if err != nil {
			slog.Warn("unreadable video gallery entry", "id", id, "error", err)
			continue
		}
		archive[id] = r.videoRecord
	}
	v.videoArchive = archive
	return nil
}

func (v *imageService) dayVideosLocked(day string, loc *time.Location) []videoRecord {
	var results []videoRecord
	for _, r := range v.videoArchive {
		if r.Created.In(loc).Format(imageGalleryDate) == day {
			results = append(results, r)
		}
	}
	sort.Slice(results, func(i, j int) bool {
		a, b := results[i], results[j]
		if a.Created.Equal(b.Created) {
			return a.ID > b.ID
		}
		return a.Created.After(b.Created)
	})
	return results
}
func videoDayToken(day, zone string, clips []videoRecord) string {
	ids := make([]string, 0, len(clips))
	for _, r := range clips {
		ids = append(ids, r.ID)
	}
	return imageGalleryDeleteToken(day, zone, ids)
}

func (s *Server) videoGalleryHandler(c *gin.Context) {
	day, zone := c.Query("day"), c.DefaultQuery("tz", "UTC")
	loc, err := imageGalleryLocation(zone)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	v := s.imageService()
	v.mu.Lock()
	defer v.mu.Unlock()
	if err = v.loadVideoGalleryLocked(); err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	if day == "" {
		groups := map[string][]string{}
		for _, r := range v.videoArchive {
			key := r.Created.In(loc).Format(imageGalleryDate)
			groups[key] = append(groups[key], r.ID)
		}
		days := make([]imageGalleryDay, 0, len(groups))
		for key, ids := range groups {
			days = append(days, imageGalleryDay{Day: key, Count: len(ids), DeleteToken: imageGalleryDeleteToken(key, zone, ids)})
		}
		sort.Slice(days, func(i, j int) bool { return days[i].Day > days[j].Day })
		c.JSON(200, gin.H{"days": days, "today": time.Now().In(loc).Format(imageGalleryDate)})
		return
	}
	if _, err = time.Parse(imageGalleryDate, day); err != nil {
		c.JSON(400, gin.H{"error": "dia inválido"})
		return
	}
	clips := v.dayVideosLocked(day, loc)
	if c.Request.Method == "DELETE" {
		if len(clips) == 0 || c.Query("token") != videoDayToken(day, zone, clips) {
			c.JSON(409, gin.H{"error": "Os vídeos do dia mudaram. Atualize e confirme novamente."})
			return
		}
		for _, r := range clips {
			if err := v.deleteVideoLocked(r); err != nil {
				c.JSON(409, gin.H{"error": err.Error()})
				return
			}
		}
		if err := v.persistJobsLocked(); err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		c.JSON(200, gin.H{"deleted": len(clips)})
		return
	}
	if before := c.Query("before"); before != "" {
		// The timestamp + ID cursor stays valid if an earlier page is deleted.
		stamp, id, ok := strings.Cut(before, "|")
		when, err := time.Parse(time.RFC3339Nano, stamp)
		if !ok || err != nil || !imageSafeID(id) {
			c.JSON(400, gin.H{"error": "cursor inválido"})
			return
		}
		start := sort.Search(len(clips), func(i int) bool {
			return clips[i].Created.Before(when) || clips[i].Created.Equal(when) && clips[i].ID < id
		})
		clips = clips[start:]
	}
	next := ""
	if len(clips) > imageGalleryPageSize {
		clips = clips[:imageGalleryPageSize]
		last := clips[len(clips)-1]
		next = last.Created.Format(time.RFC3339Nano) + "|" + last.ID
	}
	if clips == nil {
		clips = []videoRecord{}
	}
	c.JSON(200, gin.H{"videos": clips, "next": next})
}

func openVideoRoot(r videoRecord) (*os.Root, error) {
	root, err := os.OpenRoot(filepath.Dir(filepath.Dir(r.Path)))
	if err != nil {
		return nil, err
	}
	info, err := root.Lstat(r.ID)
	if err == nil && (!info.IsDir() || info.Mode()&os.ModeSymlink != 0) {
		err = errors.New("pasta de vídeo inválida")
	}
	if err != nil {
		root.Close()
		return nil, err
	}
	return root, nil
}
func (v *imageService) deleteVideoLocked(r videoRecord) error {
	for _, j := range v.jobs {
		if j.ID == r.ID && (j.executing || !imageTerminal(j.Status)) {
			return errors.New("aguarde o encerramento da geração")
		}
	}
	record := v.videoRecordPath(r.ID)
	pending := filepath.Join(v.root, "video-records", r.ID+".removing")
	if err := os.Rename(record, pending); err != nil {
		return err
	}
	if err := removeVideoOutput(r); err != nil {
		return errors.Join(err, os.Rename(pending, record))
	}
	// The rename records the confirmed deletion even if the server stops
	// between removing its WebM and committing the queue history.
	cleanup := os.Remove(pending)
	delete(v.videoArchive, r.ID)
	v.videoRevision++
	for _, j := range v.jobs {
		if j.ID == r.ID {
			j.VideoFile = nil
			j.HistoryCleared = true
			break
		}
	}
	return cleanup
}

func removeVideoOutput(r videoRecord) error {
	root, err := openVideoRoot(r)
	if err == nil {
		// Remove only the generated file. Other files placed here by the user survive.
		rel := filepath.Join(r.ID, "video.webm")
		err = root.Remove(rel)
		if err == nil || errors.Is(err, fs.ErrNotExist) {
			_ = root.Remove(r.ID)
			err = nil
		}
		root.Close()
	}
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("excluir vídeo: %w", err)
	}
	return nil
}
func (s *Server) videoDetailHandler(c *gin.Context) {
	v := s.imageService()
	id := c.Param("id")
	if c.Request.Method == "DELETE" {
		v.mu.Lock()
		defer v.mu.Unlock()
	}
	r, err := v.readVideo(id)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			c.Status(404)
		} else {
			c.JSON(500, gin.H{"error": err.Error()})
		}
		return
	}
	if c.Request.Method == "DELETE" {
		if err = v.deleteVideoLocked(r.videoRecord); err != nil {
			c.JSON(409, gin.H{"error": err.Error()})
			return
		}
		if err := v.persistJobsLocked(); err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		c.JSON(200, gin.H{"deleted": true})
		return
	}
	if r.Details == nil {
		r.Details = legacyVideoDetails(r.videoRecord)
	}
	c.JSON(200, r)
}
func (s *Server) videoPosterHandler(c *gin.Context) {
	v := s.imageService()
	r, err := v.readVideo(c.Param("id"))
	if err != nil {
		c.Status(404)
		return
	}
	root, err := openVideoRoot(r.videoRecord)
	if err != nil {
		c.Status(404)
		return
	}
	defer root.Close()
	f, err := root.Open(filepath.Join(r.ID, "video.webm"))
	if err != nil {
		c.Status(404)
		return
	}
	defer f.Close()
	data, err := imagegen.MiddleVideoWebP(f, r.Frames)
	if err != nil {
		c.JSON(422, gin.H{"error": err.Error()})
		return
	}
	c.Header("Cache-Control", "private, max-age=86400, immutable")
	c.Data(http.StatusOK, "image/webp", data)
}

func legacyVideoDetails(r videoRecord) *videoDetails {
	root, err := openVideoRoot(r)
	if err != nil {
		return nil
	}
	defer root.Close()
	f, err := root.Open(filepath.Join(r.ID, "video.webm"))
	if err != nil {
		return nil
	}
	defer f.Close()
	b, err := imagegen.EmbeddedVideoParameters(f, r.Frames)
	if err != nil {
		return nil
	}
	var old struct {
		Prompt   string `json:"prompt"`
		Negative string `json:"negative_prompt"`
		Width    int    `json:"width"`
		Height   int    `json:"height"`
		Models   struct {
			Diffusion string `json:"diffusion_model"`
		} `json:"models"`
		Sample struct {
			Steps     int    `json:"sample_steps"`
			Method    string `json:"sample_method"`
			Scheduler string `json:"scheduler"`
			Guidance  struct {
				CFG float64 `json:"txt_cfg"`
			} `json:"guidance"`
		} `json:"sample_params"`
	}
	if json.Unmarshal(b, &old) != nil || old.Width <= 0 || old.Height <= 0 {
		return nil
	}
	return &videoDetails{Prompt: old.Prompt, Negative: old.Negative, Width: old.Width, Height: old.Height, Model: filepath.Base(old.Models.Diffusion), Steps: old.Sample.Steps, CFG: old.Sample.Guidance.CFG, Sampler: old.Sample.Method, Scheduler: old.Sample.Scheduler}
}
