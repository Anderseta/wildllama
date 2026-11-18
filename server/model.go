package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

type modelLayer struct {
	manifest.Layer
	GGUF           *gguf.Metadata
	parameterCount uint64
	splitFile      string
	splitLayers    []manifest.Layer
}

func parseFromModel(ctx context.Context, name model.Name, fn func(api.ProgressResponse)) ([]*modelLayer, model.ConfigV2, error) {
	var config model.ConfigV2
	m, err := manifest.ParseNamedManifest(name)
	local := name.LocalName()
	if errors.Is(err, os.ErrNotExist) && !local.EqualFold(name) {
		// Pulls store the model under its host-free name. Reuse it only
		// when its recorded registry agrees with the requested base.
		cached, localErr := manifest.ParseNamedManifest(local)
		switch {
		case localErr == nil:
			origin := manifest.ModelOrigin(local)
			if origin == "" || !model.ParseName(origin).EqualFold(name) {
				return nil, config, fmt.Errorf("local model %s already exists with a different or unknown origin; select the local base or pull the requested registry explicitly", local.DisplayShortest())
			}
			m, err = cached, nil
		case !errors.Is(localErr, os.ErrNotExist):
			return nil, config, localErr
		}
	}
	switch {
	case errors.Is(err, os.ErrNotExist):
		if err := PullModel(ctx, name.String(), "", &registryOptions{}, func(progress api.ProgressResponse) {
			if progress.Status != "success" {
				fn(progress)
			}
		}); err != nil {
			return nil, config, err
		}

		m, err = manifest.ParseNamedManifest(local)
		if err != nil {
			return nil, config, err
		}
	case err != nil:
		return nil, config, err
	}

	if m.Config.Digest == "" {
		return nil, config, fmt.Errorf("model %s is missing its config", name.DisplayShortest())
	}
	configFile, err := m.Config.Open()
	if err != nil {
		return nil, config, fmt.Errorf("open config for %s: %w", name.DisplayShortest(), err)
	}
	if err := json.NewDecoder(configFile).Decode(&config); err != nil {
		configFile.Close()
		return nil, config, fmt.Errorf("decode config for %s: %w", name.DisplayShortest(), err)
	}
	if err := configFile.Close(); err != nil {
		return nil, config, fmt.Errorf("close config for %s: %w", name.DisplayShortest(), err)
	}

	var layers []*modelLayer
	for _, srcLayer := range m.Layers {
		layer, err := manifest.NewLayerFromLayer(srcLayer.Digest, srcLayer.MediaType, name.DisplayShortest())
		if err != nil {
			return nil, config, err
		}
		layer.Name = srcLayer.Name

		if layer.MediaType == "application/vnd.ollama.image.adapter" {
			slog.Warn("LoRA adapters are deprecated; the adapter layer is carried over but new adapters cannot be created", "model", name.DisplayShortest(), "digest", layer.Digest)
		}
		switch layer.MediaType {
		case "application/vnd.ollama.image.model",
			"application/vnd.ollama.image.projector",
			"application/vnd.ollama.image.adapter",
			manifest.MediaTypeImageDraft:
			blobpath, err := manifest.BlobsPath(layer.Digest)
			if err != nil {
				return nil, config, err
			}

			metadata, err := gguf.ReadFileMetadata(blobpath, 1)
			if err != nil {
				return nil, config, err
			}

			layers = append(layers, &modelLayer{
				Layer:          layer,
				GGUF:           metadata,
				parameterCount: metadata.ParameterCount(),
			})
		default:
			layers = append(layers, &modelLayer{Layer: layer})
		}
	}

	layers, err = groupManifestSplitGGUFLayers(layers)
	return layers, config, err
}

func detectChatTemplate(layers []*modelLayer) ([]*modelLayer, error) {
	for _, layer := range layers {
		if layer.GGUF == nil {
			continue
		}
		chatTemplate := layer.GGUF.ChatTemplate()
		if chatTemplate == "" {
			continue
		}

		t, err := template.Named(chatTemplate)
		if err != nil {
			slog.Debug("template detection", "error", err, "template", chatTemplate)
			return layers, nil
		}

		templateLayer, err := manifest.NewLayer(t.Reader(), "application/vnd.ollama.image.template")
		if err != nil {
			return nil, err
		}
		templateLayer.Status = fmt.Sprintf("using autodetected template %s", t.Name)
		layers = append(layers, &modelLayer{Layer: templateLayer})

		if t.Parameters != nil {
			var b bytes.Buffer
			if err := json.NewEncoder(&b).Encode(t.Parameters); err != nil {
				return nil, err
			}

			paramsLayer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.params")
			if err != nil {
				return nil, err
			}
			layers = append(layers, &modelLayer{Layer: paramsLayer})
		}
		return layers, nil
	}

	return layers, nil
}
