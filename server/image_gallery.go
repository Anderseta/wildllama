package server

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"net/http"
	"slices"
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
	Day         string `json:"day"`
	Count       int    `json:"count"`
	DeleteToken string `json:"delete_token"`
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
	return stamp, id, err == nil && imageSafeID(job) && validImageOutputFilename(filename)
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

func imageGalleryLocation(zone string) (*time.Location, error) {
	if len(zone) > 100 || zone == "Local" {
		return nil, errors.New("invalid gallery timezone")
	}
	return time.LoadLocation(zone)
}

func imageGalleryDayKey(stamp int64, location *time.Location) string {
	if stamp == 0 {
		return "undated"
	}
	return time.UnixMilli(stamp).In(location).Format(imageGalleryDate)
}

func imageGalleryDeleteToken(day, zone string, ids []string) string {
	slices.Sort(ids)
	digest := sha256.New()
	digest.Write([]byte(day + "\x00" + zone))
	for _, id := range ids {
		digest.Write([]byte("\x00" + id))
	}
	return hex.EncodeToString(digest.Sum(nil))
}

func (s *Server) imageGalleryHandler(c *gin.Context) {
	zone := c.DefaultQuery("tz", "UTC")
	location, err := imageGalleryLocation(zone)
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
			ids := map[string][]string{}
			for rows.Next() {
				var id string
				var stamp int64
				if err := rows.Scan(&id, &stamp); err != nil {
					return nil, err
				}
				if _, ok := v.visibleGalleryAssetLocked(id); !ok {
					continue
				}
				key := imageGalleryDayKey(stamp, location)
				index, ok := byDay[key]
				if !ok {
					index = len(days)
					byDay[key] = index
					days = append(days, imageGalleryDay{Day: key})
				}
				days[index].Count++
				ids[key] = append(ids[key], id)
			}
			for i := range days {
				days[i].DeleteToken = imageGalleryDeleteToken(days[i].Day, zone, ids[days[i].Day])
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

func (s *Server) imageGalleryDayDeleteHandler(c *gin.Context) {
	var request struct {
		Day   string `json:"day"`
		Zone  string `json:"tz"`
		Token string `json:"delete_token"`
		Count int    `json:"count"`
	}
	if !imageBind(c, &request) {
		return
	}
	location, zoneErr := imageGalleryLocation(request.Zone)
	date, dateErr := time.Parse(imageGalleryDate, request.Day)
	_, tokenErr := hex.DecodeString(request.Token)
	if zoneErr != nil || request.Zone == "" || request.Count < 1 || len(request.Token) != 64 || tokenErr != nil ||
		(request.Day != "undated" && (dateErr != nil || date.Format(imageGalleryDate) != request.Day)) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid gallery day selection"})
		return
	}
	v := s.imageService()
	v.mu.Lock()
	defer v.mu.Unlock()
	var prior []string
	for id, asset := range v.assetDeletions {
		if asset.DeletionBatch == request.Token {
			prior = append(prior, id)
		}
	}
	if len(prior) > 0 {
		// Recover a partial filesystem write or a lost acknowledgement without
		// including new images that may since have appeared on this day.
		imageAssetDeletionResponse(c, prior, v.finishImageAssetDeletionLocked())
		return
	}
	if v.assetCatalogError != nil {
		imageAssetDeletionResponse(c, nil, v.assetCatalogError)
		return
	}
	db, err := v.openImageCatalog()
	if err == nil {
		err = v.prepareImageGalleryLocked(db)
		db.Close()
	}
	if err != nil {
		imageAssetDeletionResponse(c, nil, err)
		return
	}
	ids := []string{}
	assets := []imageAssetRecord{}
	for id := range v.assets {
		if asset, ok := v.visibleGalleryAssetLocked(id); ok && imageGalleryDayKey(asset.GeneratedAtMS, location) == request.Day {
			ids = append(ids, id)
			assets = append(assets, asset)
		}
	}
	if len(ids) != request.Count || imageGalleryDeleteToken(request.Day, request.Zone, ids) != request.Token {
		c.JSON(http.StatusConflict, gin.H{"error": "As imagens deste dia mudaram. Atualize a galeria e confirme novamente."})
		return
	}
	removed, err := v.deleteImageAssetsLocked(assets, request.Token)
	imageAssetDeletionResponse(c, removed, err)
}
