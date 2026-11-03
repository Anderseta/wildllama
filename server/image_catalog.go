package server

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	_ "github.com/mattn/go-sqlite3"
)

// The old JSON snapshot is left untouched for recovery/downgrade. Once the
// catalog exists its committed rows are authoritative, including deletions.
// Every operation closes its connection, so neither shutdown nor a portable
// installation move can leave a catalog handle behind.
func (v *imageService) openImageCatalog() (*sql.DB, error) {
	path := filepath.Join(v.root, "assets.sqlite3")
	if err := imageWriteAllowed(path); err != nil {
		return nil, err
	}
	if info, err := os.Lstat(path); err == nil && !info.Mode().IsRegular() {
		return nil, errors.New("image catalog is not a regular file")
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if err := os.MkdirAll(v.root, 0700); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite3", path+"?_busy_timeout=1000&_journal_mode=WAL&_synchronous=FULL")
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err = db.Exec(`CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, record BLOB NOT NULL); CREATE TABLE IF NOT EXISTS catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);`); err != nil {
		db.Close()
		return nil, err
	}
	return db, nil
}

func (v *imageService) loadImageCatalog() error {
	if _, err := os.Lstat(filepath.Join(v.root, "assets.sqlite3")); errors.Is(err, os.ErrNotExist) {
		return nil
	} else if err != nil {
		return err
	}
	db, err := v.openImageCatalog()
	if err != nil {
		return err
	}
	defer db.Close()
	var migrated string
	if err := db.QueryRow(`SELECT value FROM catalog_meta WHERE key='json_migrated'`).Scan(&migrated); errors.Is(err, sql.ErrNoRows) {
		return nil
	} else if err != nil {
		return err
	}
	rows, err := db.Query(`SELECT id, record FROM assets`)
	if err != nil {
		return err
	}
	defer rows.Close()
	assets := map[string]imageAssetRecord{}
	for rows.Next() {
		var id string
		var data []byte
		var record imageAssetRecord
		if err = rows.Scan(&id, &data); err != nil {
			return err
		}
		if err = json.Unmarshal(data, &record); err != nil || record.AssetID != id {
			return fmt.Errorf("invalid image catalog record %q", id)
		}
		assets[id] = record
	}
	if err := rows.Err(); err != nil {
		return err
	}
	v.assets = assets
	v.assetCatalogError = nil
	return nil
}

// Called under v.mu. Transactions change only the affected records after the
// first migration, rather than re-encoding/fsyncing the entire asset history.
func (v *imageService) persistImageAssetsLocked(changed map[string]imageAssetRecord, removed []string) error {
	return v.commitImageAssetsLocked(changed, removed, false)
}

func (v *imageService) syncImageAssetsLocked() error {
	return v.commitImageAssetsLocked(v.assets, nil, true)
}

func (v *imageService) commitImageAssetsLocked(changed map[string]imageAssetRecord, removed []string, reconcile bool) error {
	if v.assetCatalogError != nil {
		return v.assetCatalogError
	}
	db, err := v.openImageCatalog()
	if err != nil {
		return err
	}
	defer db.Close()
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if reconcile {
		// Session deletion may already have unlinked files before an earlier
		// metadata commit failed. Reconcile persisted identities as well, so a
		// retry cannot leave stale rows simply because the map was updated.
		rows, err := tx.Query(`SELECT id FROM assets`)
		if err != nil {
			return err
		}
		for rows.Next() {
			var id string
			if err = rows.Scan(&id); err != nil {
				rows.Close()
				return err
			}
			if _, keep := v.assets[id]; !keep {
				removed = append(removed, id)
			}
		}
		if err = rows.Err(); err != nil {
			rows.Close()
			return err
		}
		rows.Close()
	}
	var migrated string
	if err = tx.QueryRow(`SELECT value FROM catalog_meta WHERE key='json_migrated'`).Scan(&migrated); errors.Is(err, sql.ErrNoRows) {
		changed = v.assets
	} else if err != nil {
		return err
	}
	for id, asset := range changed {
		data, err := json.Marshal(asset)
		if err != nil {
			return err
		}
		if _, err = tx.Exec(`INSERT INTO assets(id,record) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET record=excluded.record`, id, data); err != nil {
			return err
		}
	}
	for _, id := range removed {
		if _, err = tx.Exec(`DELETE FROM assets WHERE id=?`, id); err != nil {
			return err
		}
	}
	if _, err = tx.Exec(`INSERT OR REPLACE INTO catalog_meta(key,value) VALUES('json_migrated','1')`); err != nil {
		return err
	}
	return tx.Commit()
}

func (v *imageService) persistImageJobStateLocked(job *imageJob) {
	job.PersistenceError = ""
	path := filepath.Join(v.root, "job-state", job.ID+".json")
	if err := v.persistJobsLocked(); err != nil {
		job.PersistenceError = "Image task state could not be saved to history: " + err.Error()
		// A sharing violation on the main history must not turn a completed
		// output into an interrupted job after restart. This small independent
		// recovery record is also checked; an out-of-space failure stays visible.
		if backupErr := writeImageJSON(path, job); backupErr != nil {
			job.PersistenceError += "; recovery record failed: " + backupErr.Error()
		}
	} else {
		_ = os.Remove(path)
	}
}

func (v *imageService) recoverImageJobStates() {
	for _, job := range v.jobs {
		if !imageSafeID(job.ID) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(v.root, "job-state", job.ID+".json"))
		if err != nil {
			continue
		}
		var recovered imageJob
		if json.Unmarshal(data, &recovered) == nil && recovered.ID == job.ID && imageTerminal(recovered.Status) && recovered.FinishedAt != nil && (job.Status == "interrupted" || job.FinishedAt == nil || !recovered.FinishedAt.Before(*job.FinishedAt)) {
			*job = recovered
		}
	}
}
