package imagegen

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const WanI2VPackageID = "wan-rapid-v10-i2v-fp8"

func wanComponents() []Component {
	const comfy = "Comfy-Org/Wan-2.1"
	const comfyRev = "d6258323e18f14c60d37899da97ca555f3118259"
	const encoder = "rzgar/NSFW-Wan-UMT5-XXL-V2"
	const encoderRev = "f89480ab2f3ed3da9561ca886a0f2cc25f90bd94"
	return []Component{
		{Role: "diffusion", Repo: "desirel/WAN2.2-14B-Rapid-AllInOne-GGUF-NSFW-v10", Revision: "4502c7f3de51fd728d851b37eda262f2be2a5628", Filename: "wan2.2-i2v-rapid-aio-v10-nsfw-Q6_K.gguf", Size: 13741647616, SHA256: "0b0180e2fb46e8ded9af200c7670fedc5eacf5ad679d9b63e177bc56fab8d880"},
		{Role: "text_encoder", Repo: encoder, Revision: encoderRev, Filename: "encoders/nsfw_wan_umt5_xxl_v2_fp8_e4m3fn_scaled.safetensors", Size: 6741537445, SHA256: "8da6e259cfeb986dcff327364146089711c989c546d81cedd70c3e87e9cfee26"},
		{Role: "wan_vocabulary", Repo: encoder, Revision: encoderRev, Filename: "new_tokens_v4.json", Size: 10752, SHA256: "992dd94a5a0a89b724c0c745f43958e6bc9fcf351c631149649242ade96b647d"},
		{Role: "vision_encoder", Repo: "olivv-cs/wan_nsfw_clip_vision_h", Revision: "4003d23664991a62f59351224a63e527fc356190", Filename: "wan21NSFWClipVisionH_v10.safetensors", Size: 3944517804, SHA256: "be3da981f2e6a94700aeefb49fb9d84cba724ace7ea0ced110ccda95a5b2fc56"},
		{Role: "vae", Repo: comfy, Revision: comfyRev, Filename: "split_files/vae/wan_2.1_vae.safetensors", Size: 253815318, SHA256: "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b"},
	}
}

func validateWanVocabulary(path string) error {
	if !isTokenizerPath(path) {
		return errors.New("expected an absolute UMT5 vocabulary JSON path")
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
	if !info.Mode().IsRegular() || info.Size() < 4 || info.Size() > 1<<20 {
		return errors.New("invalid UMT5 vocabulary size")
	}
	var words []string
	if err := json.NewDecoder(f).Decode(&words); err != nil {
		return err
	}
	if len(words) == 0 {
		return errors.New("empty UMT5 vocabulary")
	}
	seen := make(map[string]bool, len(words))
	for _, word := range words {
		if word == "" || len(word) > 512 || strings.ContainsRune(word, 0) || seen[word] {
			return errors.New("invalid or duplicate UMT5 token")
		}
		seen[word] = true
	}
	return nil
}

func InstallWanI2V(ctx context.Context, modelsDir string, configured map[string]string, progress func(DownloadProgress)) (map[string]string, error) {
	return downloadWan(ctx, modelsDir, configured, wanComponents(), downloadClient, "https://huggingface.co", progress)
}

func downloadWan(ctx context.Context, modelsDir string, configured map[string]string, components []Component, client *http.Client, base string, progress func(DownloadProgress)) (map[string]string, error) {
	if err := validateWritePath(modelsDir); err != nil {
		return nil, err
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-installSlot:
	}
	defer func() { installSlot <- struct{}{} }()
	dir := filepath.Join(modelsDir, "videos", WanI2VPackageID)
	if err := validateWritePath(dir); err != nil {
		return nil, err
	}
	paths := make(map[string]string, len(components))
	for i, c := range components {
		target := filepath.Join(dir, filepath.Base(c.Filename))
		report := func(p DownloadProgress) {
			p.FileIndex, p.FileCount, p.Filename, p.Role = i+1, len(components), filepath.Base(c.Filename), c.Role
			if progress != nil {
				progress(p)
			}
		}
		if existing := configured[c.Role]; filepath.IsAbs(existing) && !sameComponentPath(existing, target) {
			if info, err := os.Stat(existing); err == nil && info.Mode().IsRegular() && info.Size() == c.Size {
				if err := verifyFile(ctx, existing, c.Size, c.SHA256, func(n int64) {
					report(DownloadProgress{Phase: "verifying", Message: "Verificando componente já configurado", Completed: n, Total: c.Size})
				}); err == nil {
					paths[c.Role] = existing
					continue
				}
				if ctx.Err() != nil {
					return nil, ctx.Err()
				}
			}
		}
		if err := downloadComponent(ctx, client, base, InstalledComponent{Component: c, Path: target}, report); err != nil {
			return nil, fmt.Errorf("%s: %w", filepath.Base(c.Filename), err)
		}
		paths[c.Role] = target
	}
	return paths, nil
}
