package server

import (
	"database/sql"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"
	_ "time/tzdata"

	"github.com/gin-gonic/gin"
)

const imageGalleryPageSize = 20
const imageGalleryDate = "2006-01-02"
const imageGalleryTime = "CAST(COALESCE(json_extract(record, '$.generated_at_ms'), 0) AS INTEGER)"
const imageGalleryOutputs = "COALESCE(json_extract(record, '$.input'), 0) = 0 AND COALESCE(json_extract(record, '$.deletion_completed'), 0) = 0"

var errInvalidGalleryCursor = errors.New("invalid gallery cursor")

type imageGalleryDay struct {
	Day   string `json:"day"`
	Count int    `json:"count"`
}

type imageGalleryItem struct {
	imageAsset
	ModelID string `json:"model_id,omitempty"`
}

// Open the gallery index only when browsing. Jobs remain independent of the
// one-time migration, and historical timestamps survive pruning jobs.json.
func (v *imageService) prepareImageGalleryLocked(db *sql.DB) error {
	if v.galleryIndexed {
		return nil
	}
	var version string
	err := db.QueryRow("SELECT value FROM catalog_meta WHERE key='gallery_dates_v1'").Scan(&version)
	if errors.Is(err, sql.ErrNoRows) {
		jobDates := make(map[string]int64, len(v.jobs))
		for _, job := range v.jobs {
			date := job.CreatedAt
			if job.FinishedAt != nil && job.Status != "interrupted" {
				date = *job.FinishedAt
			}
			if !date.IsZero() {
				jobDates[job.ID] = date.UnixMilli()
			}
		}
		changed := map[string]imageAssetRecord{}
		for id, asset := range v.assets {
			if asset.Input || asset.DeletionCompleted || asset.GeneratedAtMS != 0 {
				continue
			}
			asset.GeneratedAtMS = jobDates[asset.JobID]
			if file, openErr := openImageAsset(asset); openErr == nil {
				if info, statErr := file.Stat(); statErr == nil {
					asset.GeneratedAtMS = info.ModTime().UnixMilli()
				}
				file.Close()
			}
			if asset.GeneratedAtMS != 0 {
				changed[id] = asset
				v.assets[id] = asset
			}
		}
		if err = v.persistImageAssetsLocked(changed, nil); err != nil {
			for id, asset := range changed {
				asset.GeneratedAtMS = 0
				v.assets[id] = asset
			}
			return err
		}
	} else if err != nil {
		return err
	}
	if _, err = db.Exec("CREATE INDEX IF NOT EXISTS gallery_by_date ON assets (" + imageGalleryTime +
		" DESC, id DESC) WHERE " + imageGalleryOutputs); err != nil {
		return err
	}
	if _, err = db.Exec("INSERT OR IGNORE INTO catalog_meta(key,value) VALUES('gallery_dates_v1','1')"); err != nil {
		return err
	}
	v.galleryIndexed = true
	return nil
}

func (v *imageService) visibleGalleryAssetLocked(id string) (imageAssetRecord, bool) {
	asset, ok := v.assets[id]
	_, deleted := v.assetDeletions[id]
	return asset, ok && !asset.Input && !asset.DeletionCompleted && !deleted && !v.deletedSessions[asset.SessionID]
}

func splitGalleryCursor(cursor string) (int64, string, bool) {
	raw, id, _ := strings.Cut(cursor, ":")
	job, filename, _ := strings.Cut(id, "/")
	stamp, err := strconv.ParseInt(raw, 10, 64)
	return stamp, id, err == nil && imageSafeID(job) && (filename == "image.png" || imageBatchOutputName(filename))
}

// A local midnight can be skipped on a daylight-saving transition. Search
// using the local civil date so listings and page boundaries agree.
func imageGalleryDayStart(date time.Time, location *time.Location) int64 {
	wanted := date.Format(imageGalleryDate)
	lo, hi := date.Add(-26*time.Hour).UnixMilli(), date.Add(26*time.Hour).UnixMilli()
	for lo < hi {
		mid := lo + (hi-lo)/2
		if time.UnixMilli(mid).In(location).Format(imageGalleryDate) < wanted {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

func (s *Server) imageGalleryHandler(c *gin.Context) {
	zone := c.DefaultQuery("tz", "UTC")
	if len(zone) > 100 || zone == "Local" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid gallery timezone"})
		return
	}
	location, err := time.LoadLocation(zone)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid gallery timezone"})
		return
	}
	day, cursor := c.Query("day"), c.Query("before")
	query := "SELECT id, " + imageGalleryTime + " FROM assets WHERE " + imageGalleryOutputs
	var args []any
	if day == "undated" {
		query += " AND " + imageGalleryTime + " = 0"
	} else if day != "" {
		date, err := time.Parse(imageGalleryDate, day)
		if err != nil || date.Format(imageGalleryDate) != day {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid gallery day"})
			return
		}
		query += " AND " + imageGalleryTime + " != 0 AND " + imageGalleryTime + " BETWEEN ? AND ?"
		args = append(args, imageGalleryDayStart(date, location), imageGalleryDayStart(date.AddDate(0, 0, 1), location)-1)
	}
	v := s.imageService()
	v.mu.Lock()
	result, err := func() (gin.H, error) {
		if v.assetCatalogError != nil {
			return nil, v.assetCatalogError
		}
		db, err := v.openImageCatalog()
		if err != nil {
			return nil, err
		}
		defer db.Close()
		if err := v.prepareImageGalleryLocked(db); err != nil {
			return nil, err
		}
		if cursor != "" && day != "" {
			// The cursor includes its sort key so removing its image doesn't
			// invalidate the next page.
			stamp, id, ok := splitGalleryCursor(cursor)
			if !ok {
				return nil, errInvalidGalleryCursor
			}
			query += " AND (" + imageGalleryTime + ", id) < (?, ?)"
			args = append(args, stamp, id)
		}
		rows, err := db.QueryContext(c.Request.Context(), query+" ORDER BY "+imageGalleryTime+" DESC, id DESC", args...)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		if day == "" {
			days := []imageGalleryDay{}
			byDay := map[string]int{}
			for rows.Next() {
				var id string
				var stamp int64
				if err := rows.Scan(&id, &stamp); err != nil {
					return nil, err
				}
				if _, ok := v.visibleGalleryAssetLocked(id); !ok {
					continue
				}
				key := "undated"
				if stamp != 0 {
					key = time.UnixMilli(stamp).In(location).Format(imageGalleryDate)
				}
				index, ok := byDay[key]
				if !ok {
					index = len(days)
					byDay[key] = index
					days = append(days, imageGalleryDay{Day: key})
				}
				days[index].Count++
			}
			return gin.H{"days": days, "today": time.Now().In(location).Format(imageGalleryDate)}, rows.Err()
		}
		items := make([]imageGalleryItem, 0, imageGalleryPageSize)
		next := ""
		for rows.Next() {
			var id string
			var stamp int64
			if err := rows.Scan(&id, &stamp); err != nil {
				return nil, err
			}
			asset, ok := v.visibleGalleryAssetLocked(id)
			if !ok {
				continue
			}
			if len(items) == imageGalleryPageSize {
				last := items[len(items)-1]
				next = strconv.FormatInt(last.GeneratedAtMS, 10) + ":" + last.AssetID
				break
			}
			item := imageGalleryItem{imageAsset: asset.imageAsset}
			item.GeneratedAtMS = stamp
			if asset.Generation != nil {
				item.ModelID = asset.Generation.ModelID
			}
			items = append(items, item)
		}
		return gin.H{"images": items, "next": next}, rows.Err()
	}()
	v.mu.Unlock()
	if errors.Is(err, errInvalidGalleryCursor) {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, result)
}
