package imagegen

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const loraManifestName = "wildimage-lora.json"

type LoRAAsset struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Path     string `json:"path"`
	Size     int64  `json:"size"`
	Ready    bool   `json:"ready"`
	Origin   string `json:"origin"`
	Error    string `json:"error,omitempty"`
	Repo     string `json:"repo"`
	Revision string `json:"revision"`
	Filename string `json:"filename"`
	SHA256   string `json:"sha256"`
}

type downloadTokenKey struct{}

// WithDownloadToken carries an ephemeral HF credential; it is never placed in
// an asset manifest, command line, progress event, or process environment.
func WithDownloadToken(ctx context.Context, token string) context.Context {
	return context.WithValue(ctx, downloadTokenKey{}, token)
}

type credentialTransport struct {
	next  http.RoundTripper
	token string
}

func (t credentialTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	clone := request.Clone(request.Context())
	clone.Header = request.Header.Clone()
	if clone.URL.Scheme == "https" && strings.EqualFold(clone.URL.Hostname(), "huggingface.co") && clone.URL.Port() == "" {
		if t.token != "" {
			clone.Header.Set("Authorization", "Bearer "+t.token)
		}
	} else {
		clone.Header.Del("Authorization")
	}
	return t.next.RoundTrip(clone)
}

// InstallLoRA downloads only an explicitly selected, immutable server-resolved
// component. Publication follows SHA256 and native-compatible header checks.
func InstallLoRA(ctx context.Context, modelsDir string, component Component, progress func(DownloadProgress)) (LoRAAsset, error) {
	client := *downloadClient
	transport := client.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	token, _ := ctx.Value(downloadTokenKey{}).(string)
	client.Transport = credentialTransport{next: transport, token: token}
	return installLoRA(ctx, modelsDir, component, progress, &client, "https://huggingface.co")
}

var repositoryID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)

func installLoRA(ctx context.Context, modelsDir string, c Component, progress func(DownloadProgress), client *http.Client, baseURL string) (LoRAAsset, error) {
	asset := LoRAAsset{Name: filepath.Base(c.Filename), Size: c.Size, Repo: c.Repo, Revision: c.Revision, Filename: c.Filename, SHA256: strings.ToLower(c.SHA256)}
	if !repositoryID.MatchString(c.Repo) || len(c.Revision) < 40 || len(c.Revision) > 64 {
		return asset, errors.New("LoRA install requires an immutable repository revision")
	}
	if _, err := hex.DecodeString(c.Revision); err != nil {
		return asset, errors.New("invalid LoRA repository revision")
	}
	digest, err := hex.DecodeString(c.SHA256)
	if err != nil || len(digest) != 32 || c.Size <= 0 {
		return asset, errors.New("LoRA install requires verified size and SHA256")
	}
	parts := strings.Split(c.Filename, "/")
	for _, part := range parts {
		if part == "" || part == "." || part == ".." || strings.ContainsAny(part, "\\\x00\r\n:") {
			return asset, errors.New("invalid LoRA repository filename")
		}
	}
	ext := strings.ToLower(filepath.Ext(c.Filename))
	if ext != ".safetensors" && ext != ".gguf" {
		return asset, errors.New("LoRA library supports safetensors and GGUF files")
	}
	if err := validateWritePath(modelsDir); err != nil {
		return asset, err
	}
	select {
	case <-ctx.Done():
		return asset, ctx.Err()
	case <-installSlot:
	}
	defer func() { installSlot <- struct{}{} }()
	library := filepath.Join(modelsDir, "loras")
	directory := filepath.Join(library, asset.SHA256)
	stage := filepath.Join(library, ".incoming-"+asset.SHA256)
	if err := validateWritePath(stage); err != nil {
		return asset, err
	}
	if err := validateWritePath(directory); err != nil {
		return asset, err
	}
	asset.ID = asset.SHA256
	asset.Path = filepath.Join(directory, "adapter"+ext)
	asset.Origin = "https://huggingface.co/" + c.Repo + "/blob/" + c.Revision + "/" + strings.Join(escapeParts(c.Filename), "/")
	// An already published identity stays in place and is revalidated. A new
	// identity remains hidden/resumable until its directory is atomically moved.
	writeDir := stage
	if info, err := os.Stat(directory); err == nil {
		if !info.IsDir() {
			return asset, errors.New("LoRA library identity is not a directory")
		}
		writeDir = directory
	} else if !errors.Is(err, os.ErrNotExist) {
		return asset, err
	}
	if err := os.MkdirAll(writeDir, 0700); err != nil {
		return asset, err
	}
	c.Role = "lora"
	c.Required = true
	component := InstalledComponent{Component: c, Path: filepath.Join(writeDir, "adapter"+ext)}
	callback := func(p DownloadProgress) {
		p.Role = "lora"
		p.Filename = c.Filename
		p.FileIndex = 1
		p.FileCount = 1
		if progress != nil {
			progress(p)
		}
	}
	if err := downloadComponent(ctx, client, baseURL, component, callback); err != nil {
		asset.Error = err.Error()
		return asset, err
	}
	inspection, err := InspectLocalComponent(component.Path, "lora")
	if err != nil {
		asset.Error = err.Error()
		return asset, err
	}
	if inspection.Compatibility != "supported" {
		asset.Error = inspection.Hint
		return asset, errors.New(inspection.Hint)
	}
	if ext == ".gguf" {
		if ok, err := ggufLoRAMetadata(component.Path); err != nil || !ok {
			if err == nil {
				err = errors.New("GGUF file does not declare adapter.type=lora")
			}
			asset.Error = err.Error()
			return asset, err
		}
	}
	asset.Ready = true
	if err := writeLoRAAsset(filepath.Join(writeDir, loraManifestName), asset); err != nil {
		return asset, err
	}
	if writeDir != directory {
		if err := os.Rename(writeDir, directory); err != nil {
			return asset, err
		}
	}
	callback(DownloadProgress{Phase: "completed", Message: "LoRA verified and available for explicit profile selection", Completed: asset.Size, Total: asset.Size})
	return asset, nil
}

func writeLoRAAsset(path string, asset LoRAAsset) error {
	data, err := json.MarshalIndent(asset, "", "  ")
	if err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".lora-record-")
	if err != nil {
		return err
	}
	temporary := file.Name()
	defer os.Remove(temporary)
	if _, err = file.Write(data); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(temporary, path)
}

func LoRALibrary(modelsDir string) ([]LoRAAsset, error) {
	library := filepath.Join(modelsDir, "loras")
	entries, err := os.ReadDir(library)
	if errors.Is(err, os.ErrNotExist) {
		return []LoRAAsset{}, nil
	}
	if err != nil {
		return nil, err
	}
	assets := []LoRAAsset{}
	for _, entry := range entries {
		if len(assets) >= 1024 {
			break
		}
		if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
			continue
		}
		file, err := os.Open(filepath.Join(library, entry.Name(), loraManifestName))
		if err != nil {
			continue
		}
		var asset LoRAAsset
		err = json.NewDecoder(io.LimitReader(file, 65536)).Decode(&asset)
		file.Close()
		if err != nil {
			continue
		}
		if asset.ID != entry.Name() || len(asset.ID) != 64 {
			continue
		}
		expected := filepath.Join(library, entry.Name(), "adapter"+strings.ToLower(filepath.Ext(asset.Filename)))
		if filepath.Clean(expected) != filepath.Clean(asset.Path) {
			continue
		}
		asset.Ready = false
		asset.Error = ""
		if info, err := os.Stat(asset.Path); err != nil || !info.Mode().IsRegular() || info.Size() != asset.Size {
			asset.Error = "LoRA file is missing or has changed size"
		} else if inspection, err := InspectLocalComponent(asset.Path, "lora"); err != nil {
			asset.Error = err.Error()
		} else if inspection.Compatibility != "supported" {
			asset.Error = inspection.Hint
		} else {
			asset.Ready = true
		}
		if asset.Ready && strings.EqualFold(filepath.Ext(asset.Path), ".gguf") {
			if ok, err := ggufLoRAMetadata(asset.Path); err != nil || !ok {
				asset.Ready = false
				asset.Error = "GGUF LoRA adapter metadata is invalid"
			}
		}
		assets = append(assets, asset)
	}
	return assets, nil
}

func ggufLoRAMetadata(path string) (bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer file.Close()
	r := io.LimitReader(file, 16<<20)
	var header struct {
		Magic   [4]byte
		Version uint32
		Tensors uint64
		Keys    uint64
	}
	if err := binary.Read(r, binary.LittleEndian, &header); err != nil {
		return false, err
	}
	if string(header.Magic[:]) != "GGUF" || (header.Version != 2 && header.Version != 3) || header.Keys > 65536 {
		return false, errors.New("invalid GGUF adapter metadata")
	}
	readString := func() (string, error) {
		var n uint64
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return "", err
		}
		if n > 16<<20 {
			return "", errors.New("GGUF metadata string exceeds limit")
		}
		data := make([]byte, int(n))
		_, err := io.ReadFull(r, data)
		return string(data), err
	}
	var skip func(uint32) error
	skip = func(kind uint32) error {
		width := map[uint32]int64{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
		if n, ok := width[kind]; ok {
			_, err := io.CopyN(io.Discard, r, n)
			return err
		}
		if kind == 8 {
			_, err := readString()
			return err
		}
		if kind == 9 {
			var element uint32
			var count uint64
			if err := binary.Read(r, binary.LittleEndian, &element); err != nil {
				return err
			}
			if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
				return err
			}
			if count > 65536 || element == 9 {
				return errors.New("unsupported GGUF metadata array")
			}
			for range count {
				if err := skip(element); err != nil {
					return err
				}
			}
			return nil
		}
		return errors.New("unsupported GGUF metadata type")
	}
	adapter := false
	for range header.Keys {
		key, err := readString()
		if err != nil {
			return false, err
		}
		var kind uint32
		if err := binary.Read(r, binary.LittleEndian, &kind); err != nil {
			return false, err
		}
		if kind == 8 {
			value, err := readString()
			if err != nil {
				return false, err
			}
			if key == "adapter.type" && value == "lora" {
				adapter = true
			}
		} else if err := skip(kind); err != nil {
			return false, err
		}
	}
	return adapter, nil
}
