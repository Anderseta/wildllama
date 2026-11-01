package manifest

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/ollama/ollama/types/model"
)

type DownloadCleanup struct {
	RemovedFiles     int      `json:"removed_files"`
	RemovedBytes     int64    `json:"removed_bytes"`
	RetainedBlobs    int      `json:"retained_blobs"`
	DiscardedDigests []string `json:"-"`
}

var downloadPartSuffix = regexp.MustCompile(`^-(?:partial|partial-[0-9]+)$`)

// DiscardDownload removes only files recorded by this download's pin. The caller
// must exclude active downloads/imports until this operation returns.
func DiscardDownload(name model.Name) (result DownloadCleanup, err error) {
	if !name.IsFullyQualified() || !name.IsValid() {
		return result, errors.New("invalid model download name")
	}
	manifestStoreMu.Lock()
	defer manifestStoreMu.Unlock()
	pinPath := downloadPinPath(name)
	info, err := os.Lstat(pinPath)
	if errors.Is(err, os.ErrNotExist) {
		return result, nil
	}
	if err != nil {
		return result, err
	}
	if !info.Mode().IsRegular() || info.Size() > maxDownloadPinBytes {
		return result, errors.New("invalid download metadata")
	}
	file, err := os.Open(pinPath)
	if err != nil {
		return result, err
	}
	data, err := io.ReadAll(io.LimitReader(file, maxDownloadPinBytes+1))
	file.Close()
	if err != nil {
		return result, err
	}
	var pin downloadPin
	if len(data) > maxDownloadPinBytes || json.Unmarshal(data, &pin) != nil || len(pin.Digests) > 4096 || !strings.EqualFold(pin.Model, name.String()) {
		return result, errors.New("invalid download metadata")
	}
	digests := []string{}
	seen := map[string]bool{}
	for _, digest := range pin.Digests {
		if err := ValidateDigest(digest); err != nil {
			return result, err
		}
		digest = NormalizeDigest(digest)
		if !seen[digest] {
			digests = append(digests, digest)
			seen[digest] = true
		}
	}
	pin.Digests = digests
	retained, err := downloadPinnedBlobsLocked(name)
	if err != nil {
		return result, err
	}
	refs, err := namedManifestRefs(true)
	if err != nil {
		return result, err
	}
	for _, ref := range refs {
		data, _, digest, err := readVerifiedManifestLocked(ref.path, ref.root)
		if err != nil {
			return result, fmt.Errorf("cannot verify installed models: %w", err)
		}
		digests, err := referencedBlobDigestsForData(digest, data)
		if err != nil {
			return result, err
		}
		for _, digest := range digests {
			retained[digest] = struct{}{}
		}
	}
	type candidate struct {
		path  string
		bytes int64
	}
	files := map[string][]candidate{}
	for _, digest := range pin.Digests {
		if _, exists := files[digest]; exists {
			continue
		}
		if _, used := retained[digest]; used {
			result.RetainedBlobs++
			continue
		}
		blob, err := BlobsPath(digest)
		if err != nil {
			return result, err
		}
		parts, err := filepath.Glob(blob + "-partial-*")
		if err != nil {
			return result, err
		}
		paths := append([]string{blob, blob + "-partial"}, parts...)
		files[digest] = nil
		for _, path := range paths {
			if path != blob && !downloadPartSuffix.MatchString(strings.TrimPrefix(path, blob)) {
				continue
			}
			info, err := os.Lstat(path)
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			if err != nil {
				return result, err
			}
			if !info.Mode().IsRegular() {
				return result, errors.New("download artifact is not a regular file")
			}
			files[digest] = append(files[digest], candidate{path, info.Size()})
		}
	}
	for digest, candidates := range files {
		for _, candidate := range candidates {
			if err := os.Remove(candidate.path); errors.Is(err, os.ErrNotExist) {
				continue
			} else if err != nil {
				return result, err
			}
			result.RemovedFiles++
			result.RemovedBytes += candidate.bytes
		}
		result.DiscardedDigests = append(result.DiscardedDigests, digest)
	}
	if err := os.Remove(pinPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return result, err
	}
	return result, nil
}
