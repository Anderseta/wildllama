package server

import (
	"crypto/rand"
	"encoding/hex"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/imagegen"
)

const imageModelPrefixLimit = 24

var (
	imageModelWords       = regexp.MustCompile("[a-z0-9]+")
	copiedDiffusionPrefix = regexp.MustCompile("^[0-9]+-diffusion_model-")
	generatedImageName    = regexp.MustCompile("^[a-z0-9][a-z0-9-]{0,23}_[a-f0-9]{16}_(0|[1-9][0-9]{0,18})[.]png$")
)

func imageModelPrefix(profile imagegen.Profile) string {
	switch profile.ModelID {
	case imagegen.OmniFP8RecipeID:
		return "omni"
	case imagegen.DefaultRecipeID:
		return "qwen"
	case imagegen.MOSSRecipeID:
		return "moss"
	case imagegen.MiniMaxH3RecipeID:
		return "minimax-h3"
	}

	name := profile.ModelID
	if name == "" || strings.HasPrefix(name, "local-") {
		name = filepath.Base(profile.DiffusionModel)
		name = copiedDiffusionPrefix.ReplaceAllString(name, "")
		name = strings.TrimSuffix(name, filepath.Ext(name))
	}
	words := imageModelWords.FindAllString(strings.ToLower(name), -1)
	if len(words) == 0 {
		return "imagem"
	}
	if words[0] == "noct" {
		return "noct"
	}
	normalized := strings.Join(words, "-")
	return strings.TrimRight(normalized[:min(len(normalized), imageModelPrefixLimit)], "-")
}

func imageOutputFilename(profile imagegen.Profile, seed int64) string {
	var hash [8]byte
	rand.Read(hash[:])
	return imageModelPrefix(profile) + "_" + hex.EncodeToString(hash[:]) + "_" + strconv.FormatInt(seed, 10) + ".png"
}

func validImageOutputFilename(filename string) bool {
	return filename == "image.png" || imageBatchOutputName(filename) || generatedImageName.MatchString(filename)
}

func validImageAssetFilename(filename string, input bool) bool {
	if input {
		return filename == "image.png" || filename == "image.jpg" || filename == "image.webp"
	}
	return validImageOutputFilename(filename)
}
