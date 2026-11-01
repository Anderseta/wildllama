package manifest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

func modelOriginPath(name model.Name) string {
	key := sha256.Sum256([]byte(strings.ToLower(name.LocalName().String())))
	return filepath.Join(envconfig.Models(), "metadata", fmt.Sprintf("model-origin-%x.json", key))
}

// ModelOrigin is download provenance, never the local model's name.
func ModelOrigin(name model.Name) string {
	file, err := os.Open(modelOriginPath(name))
	if err != nil {
		return ""
	}
	defer file.Close()
	var record struct {
		Source string `json:"source"`
	}
	if json.NewDecoder(io.LimitReader(file, 8192)).Decode(&record) != nil {
		return ""
	}
	source := model.ParseName(record.Source)
	if !source.IsValid() || !source.LocalName().EqualFold(name.LocalName()) {
		return ""
	}
	return record.Source
}

func SetModelOrigin(name model.Name, source string) error {
	origin := model.ParseName(source)
	if !name.IsValid() || !origin.IsValid() || !origin.LocalName().EqualFold(name.LocalName()) {
		return errors.New("invalid model origin")
	}
	data, _ := json.Marshal(struct {
		Source string `json:"source"`
	}{source})
	manifestStoreMu.Lock()
	defer manifestStoreMu.Unlock()
	path := modelOriginPath(name)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".model-origin-*")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	if _, err := file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(file.Name(), path)
}

func ReleaseModelOrigin(name model.Name) error {
	manifestStoreMu.Lock()
	defer manifestStoreMu.Unlock()
	err := os.Remove(modelOriginPath(name))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

// LocalizeModelNames moves only manifest references; model blobs are untouched.
// Conflicting names are left intact and reported rather than overwritten.
func LocalizeModelNames() (int, error) {
	all, err := Manifests(false)
	if err != nil {
		return 0, err
	}
	count := 0
	for source := range all {
		target := source.LocalName()
		if source.EqualFold(target) {
			continue
		}
		data, err := ReadManifestData(source)
		if err != nil {
			return count, err
		}
		if existing, err := ReadManifestData(target); err == nil {
			if !bytes.Equal(data, existing) {
				return count, fmt.Errorf("local model name conflict: %s", target.DisplayShortest())
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return count, err
		}
		if err := SetModelOrigin(target, source.DisplayShortest()); err != nil {
			return count, err
		}
		if err := WriteManifestData(target, data); err != nil {
			return count, err
		}
		manifestStoreMu.Lock()
		err = removeNamedManifestPaths(source)
		manifestStoreMu.Unlock()
		if err != nil {
			return count, err
		}
		count++
	}
	return count, nil
}
