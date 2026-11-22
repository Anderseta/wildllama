package imagegen

import (
	"context"
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWanPackageResumesAndReusesExactLocalComponents(t *testing.T) {
	root := t.TempDir()
	ext := filepath.Join(root, "custom.gguf")
	weights := []byte("GGUF78901234567890")
	tokens := []byte(`["accelerating","zooms"]`)
	require.NoError(t, os.WriteFile(ext, weights, 0600))
	component := func(role, file string, data []byte) Component {
		return Component{Role: role, Repo: "wan/test", Revision: "revision", Filename: file, Size: int64(len(data)), SHA256: fmt.Sprintf("%x", sha256.Sum256(data))}
	}
	components := []Component{component("text_encoder", "model.gguf", weights), component("wan_vocabulary", "words.json", tokens)}
	dir := filepath.Join(root, "videos", WanI2VPackageID)
	require.NoError(t, os.MkdirAll(dir, 0700))
	require.NoError(t, os.WriteFile(filepath.Join(dir, "words.json.part"), tokens[:4], 0600))
	requests := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/wan/test/resolve/revision/words.json" || r.Header.Get("Range") != "bytes=4-" {
			http.Error(w, "unexpected request", 400)
			return
		}
		w.Header().Set("Content-Range", fmt.Sprintf("bytes 4-%d/%d", len(tokens)-1, len(tokens)))
		w.WriteHeader(206)
		_, _ = w.Write(tokens[4:])
	}))
	defer srv.Close()
	for i := 0; i < 2; i++ {
		paths, err := downloadWan(context.Background(), root, map[string]string{"text_encoder": ext}, components, srv.Client(), srv.URL, nil)
		require.NoError(t, err)
		require.Equal(t, ext, paths["text_encoder"])
		require.NoFileExists(t, filepath.Join(dir, "model.gguf"))
		require.NoError(t, validateWanVocabulary(paths["wan_vocabulary"]))
	}
	require.Equal(t, 1, requests)
	published := wanComponents()
	require.Len(t, published, 5)
	for _, c := range published {
		if c.Role == "vision_encoder" {
			require.Equal(t, "olivv-cs/wan_nsfw_clip_vision_h", c.Repo)
			require.Equal(t, int64(3944517804), c.Size)
		}
		if c.Role == "text_encoder" {
			require.Contains(t, c.Filename, "fp8_e4m3fn_scaled")
		}
	}
}
