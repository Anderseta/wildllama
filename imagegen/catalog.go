package imagegen

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

func Recipes() []Recipe {
	encoder := Component{Role: "text_encoder", Repo: "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF", Revision: "881a778e306c44920e42536def3e52c3a3ecc7cd", Filename: "Qwen-Image-2.1-Turbo-Uncensored-AD-Q4_K.gguf", SHA256: "8123f36d2b3e562deef164dffdae33bdb1d152797ad10bb630827eaa7bce795d", Size: 5287517120, Required: true}
	vae := Component{Role: "vae", Repo: "Comfy-Org/Qwen-Image-2.1", Revision: "aacf31a6d48e707f352ecba714cee0680da3479b", Filename: "vae/qwen_image_2.1_vae_bf16.safetensors", SHA256: "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9", Size: 675509688, Required: true}
	vision := Component{Role: "vision_encoder", Repo: encoder.Repo, Revision: encoder.Revision, Filename: "mmproj-Qwen-Image-2.1-Turbo-Uncensored-F16.gguf", SHA256: "d9870a1f168db3d814596216aaf35702154147b443572c7939acca1802621ac6", Size: 1159029984, Required: false}
	var result []Recipe
	for _, q := range []struct {
		tag, file, sha string
		size           int64
	}{
		{"q5", "Qwen-Image-2.1-Turbo-AD-Q5_K.gguf", "c56db41508c4c3c0d5782e40e4f153c38a1154ac9466c1c5c50a6e4bd09f6a69", 5399168736},
		{"q4", "Qwen-Image-2.1-Turbo-AD-Q4_K.gguf", "4bb73c53cbe284bbd6d69b9fdc59539c531f0389dd2aa567663d777f8130bc59", 4201694944},
	} {
		id := "qwen-image-2.1-turbo-uncensored-" + q.tag
		diffusion := Component{Role: "diffusion_model", Repo: "AtomicChat/Qwen-Image-2.1-Turbo-GGUF", Revision: "cbcb9ff83b17e1eeef693479a8d34cea355a0ee4", Filename: q.file, SHA256: q.sha, Size: q.size, Required: true}
		result = append(result, Recipe{ID: id, Name: "Qwen Image 2.1 Turbo · " + strings.ToUpper(q.tag) + " + Uncensored encoder Q4", Family: "qwen-image-2.1-turbo", Capabilities: []string{"generate", "edit"}, Components: []Component{diffusion, encoder, vae, vision}, DefaultProfile: DefaultProfile(id)})
	}
	return result
}

var bundleID = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$`)

func Catalog(modelsDir string) ([]Bundle, []Recipe, error) {
	entries, err := os.ReadDir(modelsDir)
	if errors.Is(err, os.ErrNotExist) {
		return []Bundle{}, Recipes(), nil
	}
	if err != nil {
		return nil, Recipes(), err
	}
	models := []Bundle{}
	for _, entry := range entries {
		if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		if len(models) >= 512 {
			break
		}
		manifest := filepath.Join(modelsDir, entry.Name(), ManifestName)
		if _, err := os.Stat(manifest); errors.Is(err, os.ErrNotExist) {
			continue
		}
		model, err := readBundle(manifest)
		if err != nil {
			models = append(models, Bundle{ID: entry.Name(), Name: entry.Name(), Directory: filepath.Dir(manifest), Ready: false, Missing: []string{err.Error()}})
			continue
		}
		refreshBundle(&model)
		models = append(models, model)
	}
	return models, Recipes(), nil
}

func readBundle(path string) (Bundle, error) {
	f, err := os.Open(path)
	if err != nil {
		return Bundle{}, err
	}
	defer f.Close()
	var b Bundle
	decoder := json.NewDecoder(io.LimitReader(f, 1<<20))
	if err := decoder.Decode(&b); err != nil {
		return b, fmt.Errorf("invalid image model manifest: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return b, errors.New("image manifest exceeds the size limit or contains trailing data")
	}
	if !bundleID.MatchString(b.ID) || b.ID == "." || b.ID == ".." {
		return b, errors.New("invalid image model id")
	}
	if len(b.Components) < 3 || len(b.Components) > 8 {
		return b, errors.New("image bundle must contain three required components and optional projector")
	}
	b.Directory = filepath.Dir(path)
	roles := map[string]bool{}
	for i := range b.Components {
		c := &b.Components[i]
		if c.Role != "diffusion_model" && c.Role != "text_encoder" && c.Role != "vae" && c.Role != "vision_encoder" {
			return b, errors.New("unknown image component role")
		}
		if roles[c.Role] {
			return b, errors.New("duplicate image component role")
		}
		roles[c.Role] = true
		if !filepath.IsAbs(c.Path) {
			c.Path = filepath.Join(b.Directory, c.Path)
		}
		c.Path = filepath.Clean(c.Path)
		if c.Role != "vision_encoder" {
			c.Required = true
		}
	}
	for _, role := range []string{"diffusion_model", "text_encoder", "vae"} {
		if !roles[role] {
			return b, fmt.Errorf("missing %s component", role)
		}
	}
	if b.Profile.Width == 0 {
		b.Profile = DefaultProfile(b.ID)
	}
	for i := range b.Profile.LoRAs {
		if b.Profile.LoRAs[i].Path != "" && !filepath.IsAbs(b.Profile.LoRAs[i].Path) {
			b.Profile.LoRAs[i].Path = filepath.Join(b.Directory, b.Profile.LoRAs[i].Path)
		}
	}
	if err := ValidateProfile(b.Profile); err != nil {
		return b, err
	}
	bindProfile(&b)
	return b, nil
}

func bindProfile(b *Bundle) {
	b.Profile.ModelID = b.ID
	for _, c := range b.Components {
		switch c.Role {
		case "diffusion_model":
			b.Profile.DiffusionModel = c.Path
		case "text_encoder":
			b.Profile.TextEncoder = c.Path
		case "vae":
			b.Profile.VAE = c.Path
		case "vision_encoder":
			b.Profile.VisionEncoder = c.Path
		}
	}
}
func refreshBundle(b *Bundle) {
	bindProfile(b)
	b.Ready = true
	b.Missing = []string{}
	for i := range b.Components {
		c := &b.Components[i]
		err := validateModelFile(c.Path)
		if info, statErr := os.Stat(c.Path); err == nil && (statErr != nil || c.Size > 0 && info.Size() != c.Size) {
			err = errors.New("component size differs from its manifest")
		}
		if err != nil && c.Required {
			b.Ready = false
			b.Missing = append(b.Missing, c.Role+": "+err.Error())
		}
		c.Ready = err == nil
		c.Error = ""
		c.DownloadedBytes = 0
		c.DownloadState = "missing"
		if info, statErr := os.Stat(c.Path); statErr == nil && info.Mode().IsRegular() {
			c.DownloadedBytes = info.Size()
			if c.Ready {
				c.DownloadState = "ready"
			} else {
				c.DownloadState = "invalid"
			}
		} else if info, statErr := os.Lstat(c.Path + ".part"); statErr == nil && info.Mode().IsRegular() {
			c.DownloadedBytes = info.Size()
			if info.Size() > 0 {
				c.DownloadState = "partial"
			}
			if c.Size > 0 && info.Size() == c.Size {
				c.DownloadState = "pending_verification"
			}
		}
		if err != nil {
			c.Error = err.Error()
		}
		if err != nil && c.Role == "vision_encoder" {
			b.Profile.VisionEncoder = ""
		}
	}
}
func writeBundle(path string, b Bundle) error {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".wildimage-manifest-")
	if err != nil {
		return err
	}
	temp := f.Name()
	defer os.Remove(temp)
	if _, err = f.Write(data); err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(temp, path)
}

// ImportLocal validates an explicit bundle manifest and registers immutable
// component references. Existing large weights are not copied or moved.
func ImportLocal(ctx context.Context, modelsDir, bundlePath string) (Bundle, error) {
	if err := validateWritePath(modelsDir); err != nil {
		return Bundle{}, err
	}
	if info, err := os.Stat(bundlePath); err == nil && info.IsDir() {
		bundlePath = filepath.Join(bundlePath, ManifestName)
	}
	abs, err := filepath.Abs(bundlePath)
	if err != nil {
		return Bundle{}, err
	}
	b, err := readBundle(abs)
	if err != nil {
		return b, err
	}
	for i := range b.Components {
		c := &b.Components[i]
		if err := ctx.Err(); err != nil {
			return b, err
		}
		if err := validateModelFile(c.Path); err != nil {
			if c.Required {
				return b, fmt.Errorf("%s: %w", c.Role, err)
			}
			continue
		}
		if c.SHA256 != "" {
			if err := verifyFile(ctx, c.Path, c.Size, c.SHA256, nil); err != nil {
				return b, fmt.Errorf("%s: %w", c.Role, err)
			}
		}
		info, err := os.Stat(c.Path)
		if err != nil {
			return b, err
		}
		c.Size = info.Size()
	}
	directory := filepath.Join(modelsDir, b.ID)
	if err := validateWritePath(directory); err != nil {
		return b, err
	}
	if existing, err := readBundle(filepath.Join(directory, ManifestName)); err == nil {
		same := len(existing.Components) == len(b.Components)
		for i := range existing.Components {
			if i >= len(b.Components) || existing.Components[i].Component != b.Components[i].Component || filepath.Clean(existing.Components[i].Path) != filepath.Clean(b.Components[i].Path) {
				same = false
				break
			}
		}
		if !same {
			return b, errors.New("an image model with this id already references different components")
		}
	}
	if err := os.MkdirAll(directory, 0700); err != nil {
		return b, err
	}
	b.Directory = directory
	refreshBundle(&b)
	if err := writeBundle(filepath.Join(directory, ManifestName), b); err != nil {
		return b, err
	}
	return b, nil
}

func verifyFile(ctx context.Context, path string, size int64, digest string, progress func(int64)) error {
	expected, err := hex.DecodeString(digest)
	if err != nil || len(expected) != sha256.Size {
		return errors.New("invalid SHA256 in component manifest")
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if size > 0 && info.Size() != size {
		return errors.New("component size mismatch")
	}
	hash := sha256.New()
	buffer := make([]byte, 4<<20)
	var total int64
	last := time.Now()
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, readErr := f.Read(buffer)
		if n > 0 {
			hash.Write(buffer[:n])
			total += int64(n)
			if progress != nil && time.Since(last) >= 250*time.Millisecond {
				progress(total)
				last = time.Now()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
	}
	if !strings.EqualFold(hex.EncodeToString(hash.Sum(nil)), digest) {
		return errors.New("component SHA256 mismatch")
	}
	if progress != nil {
		progress(total)
	}
	return nil
}
