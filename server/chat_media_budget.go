package server

import (
	"bytes"
	"context"
	"image"
	"io"
	"math"
	"strings"

	"github.com/ollama/ollama/api"
)

// Each candidate retains every instruction and a suffix of the conversation.
// Read the media headers once per request, even if several cut points are tried.
type chatMediaBudget struct {
	total         int64
	removedBefore []int64
}

func estimateChatMedia(ctx context.Context, m *Model, messages []api.Message, legacy bool) (chatMediaBudget, error) {
	var budget chatMediaBudget
	if m == nil || len(m.ProjectorPaths) == 0 {
		return budget, nil
	}
	var removable int64
	for i, message := range messages {
		if len(message.Images) > 0 && budget.removedBefore == nil {
			budget.removedBefore = make([]int64, len(messages))
		}
		if budget.removedBefore != nil {
			budget.removedBefore[i] = removable
		}
		for _, data := range message.Images {
			if err := ctx.Err(); err != nil {
				return chatMediaBudget{}, err
			}
			cost := 768 // Legacy templates without projector metadata.
			if !legacy || len(m.projectorMetadata) > 0 {
				config, _, err := image.DecodeConfig(io.LimitReader(bytes.NewReader(data), 128<<10))
				if err != nil {
					config = image.Config{}
				}
				cost = projectorImageTokens(m, config.Width, config.Height)
			}
			budget.total += int64(cost)
			if !promptInstruction(message.Role) {
				removable += int64(cost)
			}
		}
	}
	return budget, nil
}

func (budget chatMediaBudget) at(start int) int {
	if budget.removedBefore == nil {
		return 0
	}
	return int(min(budget.total-budget.removedBefore[start], int64(math.MaxInt)))
}

func projectorImageTokens(m *Model, width, height int) int {
	var metadata ggufMetadata
	for _, candidate := range m.projectorMetadata {
		if candidate.Valid("vision.image_size") || candidate.Valid("vision.patch_size") || candidate.Valid("vision.projector_type") {
			metadata = candidate
			break
		}
	}
	projector := strings.ToLower(metadata.String("vision.projector_type"))
	patch := int(metadata.Int("vision.patch_size"))
	if patch <= 0 {
		patch = 14
	}
	merge := int(metadata.Int("vision.spatial_merge_size"))
	if merge <= 0 {
		merge = int(metadata.Int("vision.projector.scale_factor"))
	}
	if merge <= 0 {
		merge = 1
	}
	size := int(metadata.Int("vision.image_size"))
	ceil := func(value, divisor int) int { return int(math.Ceil(float64(value) / float64(divisor))) }
	grid := func(w, h, step int) int {
		if w <= 0 || h <= 0 || step <= 0 {
			return 4096
		}
		value := math.Ceil(float64(w)/float64(step)) * math.Ceil(float64(h)/float64(step))
		return int(min(value, 2_097_152))
	}
	switch {
	case strings.Contains(projector, "qwen") && strings.Contains(projector, "vl"), projector == "ling3vl":
		if merge == 1 {
			merge = 2
		}
		step := patch * merge
		minimum, maximum := 8, 4096 // Native mtmd Qwen-VL defaults.
		if pixels := metadata.Int("vision.image_min_pixels"); pixels > 0 {
			minimum = int((pixels + int64(step*step) - 1) / int64(step*step))
		}
		if pixels := metadata.Int("vision.image_max_pixels"); pixels > 0 {
			maximum = int((pixels + int64(step*step) - 1) / int64(step*step))
		}
		return max(minimum, min(maximum, grid(width, height, step))) + 2
	case projector == "gemma3n", projector == "gemma3nv":
		return 256
	case projector == "gemma3":
		if merge == 1 {
			merge = 4
		}
		if size <= 0 {
			size = 896
		}
		return grid(size, size, patch*merge)
	case projector == "gemma4v", projector == "gemma4uv":
		if merge == 1 {
			merge = 3
		}
		return max(70, min(1120, grid(width, height, patch*merge))) + 2
	case strings.Contains(projector, "pixtral"):
		if patch == 14 {
			patch = 16
		}
		edge := int(metadata.Int("vision.preproc_image_size"))
		if edge <= 0 {
			edge = 1024
		}
		if width <= 0 || height <= 0 {
			return grid(edge, edge, patch) + ceil(edge, patch) + 2
		}
		if longest := max(width, height); longest > edge {
			width = ceil(width*edge, longest)
			height = ceil(height*edge, longest)
		}
		return grid(width, height, patch*merge) + ceil(height, patch*merge) + 2
	case size > 0:
		cost := grid(size, size, patch*merge)
		tiles := int(metadata.Int("vision.preproc_max_tiles"))
		if strings.Contains(projector, "internvl") && tiles == 0 {
			tiles = 12
		}
		if tiles > 1 {
			if width > 0 && height > 0 {
				tiles = min(tiles, max(1, ceil(width, size)*ceil(height, size)))
			}
			cost *= min(tiles+1, 257) // Include the overview image.
		}
		return min(cost+32, 2_097_152)
	default:
		// Unknown projectors get a dimension-aware conservative estimate, never
		// the old model-independent 768-token constant.
		return min(max(768, grid(width, height, patch)), 16384) + 32
	}
}
