package imagegen

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

var installSlot = func() chan struct{} { c := make(chan struct{}, 1); c <- struct{}{}; return c }()
var downloadClient = &http.Client{Timeout: 12 * time.Hour, Transport: &http.Transport{Proxy: http.ProxyFromEnvironment, ResponseHeaderTimeout: 45 * time.Second}}

func Install(ctx context.Context, modelsDir, recipeID string, includeVision bool, progress func(DownloadProgress)) (Bundle, error) {
	if err := validateWritePath(modelsDir); err != nil {
		return Bundle{}, err
	}
	select {
	case <-ctx.Done():
		return Bundle{}, ctx.Err()
	case <-installSlot:
	}
	defer func() { installSlot <- struct{}{} }()
	var recipe Recipe
	for _, candidate := range Recipes() {
		if candidate.ID == recipeID {
			recipe = candidate
			break
		}
	}
	if recipe.ID == "" {
		return Bundle{}, errors.New("unknown image pipeline recipe")
	}
	directory := filepath.Join(modelsDir, recipe.ID)
	if err := validateWritePath(directory); err != nil {
		return Bundle{}, err
	}
	if err := os.MkdirAll(directory, 0700); err != nil {
		return Bundle{}, err
	}
	b := Bundle{ID: recipe.ID, Name: recipe.Name, RecipeID: recipe.ID, Directory: directory, Profile: recipe.DefaultProfile}
	for _, c := range recipe.Components {
		if !c.Required && !includeVision {
			continue
		}
		b.Components = append(b.Components, InstalledComponent{Component: c, Path: filepath.Join(directory, c.Role, filepath.Base(c.Filename))})
	}
	bindProfile(&b)
	refreshBundle(&b)
	var remaining int64
	for _, component := range b.Components {
		if component.Ready {
			continue
		}
		needed := component.Size
		if info, err := os.Stat(component.Path + ".part"); err == nil && info.Size() >= 0 && info.Size() <= needed {
			needed -= info.Size()
		}
		if needed < 0 || needed > (1<<62)-remaining {
			return b, errors.New("invalid image pipeline download size")
		}
		remaining += needed
	}
	if err := CheckDiskSpace(ctx, directory, remaining); err != nil {
		return b, err
	}
	if err := writeBundle(filepath.Join(directory, ManifestName), b); err != nil {
		return b, err
	}
	for i, c := range b.Components {
		callback := func(v DownloadProgress) {
			v.Role = c.Role
			v.Filename = c.Filename
			v.FileIndex = i + 1
			v.FileCount = len(b.Components)
			if progress != nil {
				progress(v)
			}
		}
		if err := downloadComponent(ctx, downloadClient, "https://huggingface.co", c, callback); err != nil {
			refreshBundle(&b)
			_ = writeBundle(filepath.Join(directory, ManifestName), b)
			return b, fmt.Errorf("%s: %w", c.Role, err)
		}
		refreshBundle(&b)
		if err := writeBundle(filepath.Join(directory, ManifestName), b); err != nil {
			return b, err
		}
	}
	refreshBundle(&b)
	return b, nil
}

type receipt struct {
	SHA256   string `json:"sha256"`
	Size     int64  `json:"size"`
	Modified int64  `json:"modified"`
}

func downloadComponent(ctx context.Context, client *http.Client, base string, c InstalledComponent, progress func(DownloadProgress)) error {
	if err := validateWritePath(c.Path); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(c.Path), 0700); err != nil {
		return err
	}
	emit := func(phase, message string, n int64) {
		if progress != nil {
			progress(DownloadProgress{Phase: phase, Message: message, Completed: n, Total: c.Size})
		}
	}
	if info, err := os.Stat(c.Path); err == nil {
		var r receipt
		if data, err := os.ReadFile(c.Path + ".verified.json"); err == nil {
			_ = json.Unmarshal(data, &r)
		}
		if r.SHA256 == c.SHA256 && r.Size == c.Size && r.Modified == info.ModTime().UnixNano() && info.Size() == c.Size {
			emit("completed", "Component already verified", c.Size)
			return nil
		}
		emit("verifying", "Verifying existing component", 0)
		if err := verifyFile(ctx, c.Path, c.Size, c.SHA256, func(n int64) { emit("verifying", "Verifying existing component", n) }); err != nil {
			return fmt.Errorf("existing component: %w", err)
		}
		return writeReceipt(c)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	part := c.Path + ".part"
	var offset int64
	if info, err := os.Lstat(part); err == nil {
		if !info.Mode().IsRegular() || info.Size() > c.Size {
			return errors.New("invalid partial component size")
		}
		offset = info.Size()
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if offset < c.Size {
		if err := CheckDiskSpace(ctx, part, c.Size-offset); err != nil {
			return err
		}
		parts := strings.Split(c.Repo, "/")
		if len(parts) != 2 || c.Revision == "" {
			return errors.New("invalid pinned model repository")
		}
		address := strings.TrimRight(base, "/") + "/" + url.PathEscape(parts[0]) + "/" + url.PathEscape(parts[1]) + "/resolve/" + url.PathEscape(c.Revision) + "/" + strings.Join(escapeParts(c.Filename), "/")
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, address, nil)
		if err != nil {
			return err
		}
		if req.URL.Scheme == "https" && req.URL.Hostname() == "huggingface.co" && req.URL.Port() == "" {
			token, _ := ctx.Value(downloadTokenKey{}).(string)
			if token == "" {
				token = os.Getenv("HF_TOKEN")
			}
			if token != "" {
				req.Header.Set("Authorization", "Bearer "+token)
			}
		}
		if offset > 0 {
			req.Header.Set("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")
		}
		response, err := client.Do(req)
		if err != nil {
			var requestError *url.Error
			if errors.As(err, &requestError) {
				return fmt.Errorf("component network request failed: %w", requestError.Err)
			}
			return err
		}
		defer response.Body.Close()
		if response.StatusCode == http.StatusOK {
			offset = 0
		} else if response.StatusCode == http.StatusPartialContent {
			var start, end, total int64
			if _, err := fmt.Sscanf(response.Header.Get("Content-Range"), "bytes %d-%d/%d", &start, &end, &total); err != nil || start != offset || total != c.Size || end < start || end >= total {
				return errors.New("download server returned an inconsistent Content-Range")
			}
		} else {
			return fmt.Errorf("component download returned HTTP %d; check repository access", response.StatusCode)
		}
		file, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY, 0600)
		if err != nil {
			return err
		}
		if offset == 0 {
			if err := file.Truncate(0); err != nil {
				file.Close()
				return err
			}
		}
		if _, err = file.Seek(offset, io.SeekStart); err != nil {
			file.Close()
			return err
		}
		emit("downloading", "Downloading component", offset)
		buffer := make([]byte, 1<<20)
		total := offset
		last := time.Now()
		lastSpaceCheck := time.Now()
		for {
			n, readErr := response.Body.Read(buffer)
			if n > 0 {
				if time.Since(lastSpaceCheck) >= time.Second {
					if err := CheckDiskSpace(ctx, part, c.Size-total); err != nil {
						file.Close()
						return err
					}
					lastSpaceCheck = time.Now()
				}
				if int64(n) > c.Size-total {
					file.Close()
					return errors.New("download exceeded the verified component size")
				}
				written, err := file.Write(buffer[:n])
				total += int64(written)
				if err != nil {
					file.Close()
					return err
				}
				if time.Since(last) > 250*time.Millisecond {
					emit("downloading", "Downloading component", total)
					last = time.Now()
				}
			}
			if readErr == io.EOF {
				break
			}
			if readErr != nil {
				file.Close()
				return readErr
			}
			if err := ctx.Err(); err != nil {
				file.Close()
				return err
			}
		}
		if err := file.Sync(); err != nil {
			file.Close()
			return err
		}
		if err := file.Close(); err != nil {
			return err
		}
		if total != c.Size {
			return errors.New("download interrupted before the expected size; resume is available")
		}
	}
	emit("verifying", "Verifying component SHA256", 0)
	if err := verifyFile(ctx, part, c.Size, c.SHA256, func(n int64) { emit("verifying", "Verifying component SHA256", n) }); err != nil {
		if ctx.Err() == nil {
			_ = os.Remove(part)
		}
		return err
	}
	if err := os.Rename(part, c.Path); err != nil {
		return err
	}
	if err := validateModelFile(c.Path); err != nil {
		return err
	}
	if err := writeReceipt(c); err != nil {
		return err
	}
	emit("completed", "Component verified", c.Size)
	return nil
}
func escapeParts(path string) []string {
	parts := strings.Split(path, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return parts
}
func writeReceipt(c InstalledComponent) error {
	info, err := os.Stat(c.Path)
	if err != nil {
		return err
	}
	data, err := json.Marshal(receipt{c.SHA256, c.Size, info.ModTime().UnixNano()})
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(c.Path), ".verified-")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if _, err = f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(name, c.Path+".verified.json")
}
