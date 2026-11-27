package server

import (
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/imagegen"
	"github.com/stretchr/testify/require"
)

func TestImageOutputNames(t *testing.T) {
	for _, tc := range []struct {
		model, file, prefix string
	}{
		{imagegen.OmniFP8RecipeID, "", "omni"},
		{imagegen.DefaultRecipeID, "", "qwen"},
		{imagegen.MOSSRecipeID, "", "moss"},
		{imagegen.MiniMaxH3RecipeID, "", "minimax-h3"},
		{"local-123", "01-diffusion_model-Noct-Q.Q8.gguf", "noct"},
		{"local-456", "01-diffusion_model-Example Dream 3.gguf", "example-dream-3"},
		{"next-image-xl-vit-quantized", "", "next-image-xl-vit-quanti"},
		{"", "猫.gguf", "imagem"},
	} {
		t.Run(tc.prefix, func(t *testing.T) {
			profile := imagegen.Profile{ModelID: tc.model, DiffusionModel: filepath.Join("models", tc.file)}
			filename := imageOutputFilename(profile, math.MaxInt64)
			require.True(t, strings.HasPrefix(filename, tc.prefix+"_"), filename)
			require.True(t, strings.HasSuffix(filename, "_9223372036854775807.png"))
			require.NotEqual(t, filename, imageOutputFilename(profile, math.MaxInt64))
			require.True(t, validImageAssetFilename(filename, false))
			require.False(t, validImageAssetFilename(filename, true))
			_, id, ok := splitGalleryCursor("123:job/" + filename)
			require.True(t, ok)
			require.Equal(t, "job/"+filename, id)
			for _, invalid := range []string{"../" + filename, "subdir/" + filename, "drive:" + filename, filename + ".exe"} {
				require.False(t, validImageAssetFilename(invalid, false))
			}
		})
	}
	for _, filename := range []string{"image.png", "image-2.png", "image-8.png"} {
		require.True(t, validImageAssetFilename(filename, false))
	}
}
