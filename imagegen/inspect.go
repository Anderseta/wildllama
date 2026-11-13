package imagegen

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/fs/gguf"
)

type tensorDescriptor struct {
	DType   string  `json:"dtype"`
	Shape   []int64 `json:"shape"`
	Offsets []int64 `json:"data_offsets"`
}

// InspectLocalComponent reads bounded headers and tiny quantization metadata,
// never tensor weight payloads. Compatibility describes the pinned runner's
// recognized storage/key layout, not an empirical full-model generation test.
func InspectLocalComponent(path, role string) (LocalInspection, error) {
	result := LocalInspection{Path: path, Role: role, Compatibility: "unsupported", DTypes: map[string]int{}}
	if role != "diffusion_model" && role != "text_encoder" && role != "vae" && role != "vision_encoder" && role != "lora" {
		return result, errors.New("unknown image component role")
	}
	if err := validateModelFile(path); err != nil {
		return result, err
	}
	f, err := os.Open(path)
	if err != nil {
		return result, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return result, err
	}
	result.Size = info.Size()
	if strings.EqualFold(filepath.Ext(path), ".gguf") {
		result.Format = "GGUF"
		metadata, err := gguf.ReadFileMetadata(path, 32)
		if err != nil {
			return result, fmt.Errorf("invalid image GGUF metadata: %w", err)
		}
		patch, merger := false, false
		for _, tensor := range metadata.TensorInfos() {
			result.TensorCount++
			result.DTypes[fmt.Sprint(tensor.Type)]++
			if strings.Contains(tensor.Name, "visual.") {
				patch = patch || strings.Contains(tensor.Name, "patch_embed")
				merger = merger || strings.Contains(tensor.Name, "merger.linear_fc2") || strings.Contains(tensor.Name, "merger.mlp.2")
			}
		}
		result.SupportsEditing = role == "text_encoder" && patch && merger
		result.Compatibility = "supported"
		result.Hint = "GGUF container recognized; the runner validates architecture and tensor shapes when used"
		return result, nil
	}
	var prefix [8]byte
	if _, err := io.ReadFull(f, prefix[:]); err != nil {
		return result, err
	}
	size := binary.LittleEndian.Uint64(prefix[:])
	if size > 16<<20 {
		return result, errors.New("safetensors metadata header exceeds 16 MiB")
	}
	header := make([]byte, int(size))
	if _, err := io.ReadFull(f, header); err != nil {
		return result, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(header, &raw); err != nil {
		return result, err
	}
	if len(raw) > 65536 {
		return result, errors.New("too many safetensors metadata entries")
	}
	tensors := map[string]tensorDescriptor{}
	for name, value := range raw {
		if name == "__metadata__" {
			continue
		}
		var tensor tensorDescriptor
		if err := json.Unmarshal(value, &tensor); err != nil {
			return result, fmt.Errorf("invalid tensor %s: %w", name, err)
		}
		if len(tensor.Offsets) != 2 || tensor.Offsets[0] < 0 || tensor.Offsets[1] < tensor.Offsets[0] || tensor.Offsets[1] > info.Size()-8-int64(size) {
			return result, errors.New("tensor offsets exceed local file")
		}
		for _, dimension := range tensor.Shape {
			if dimension < 0 {
				return result, errors.New("negative tensor dimension")
			}
		}
		tensors[name] = tensor
		result.DTypes[tensor.DType]++
		result.TensorCount++
	}
	if len(tensors) == 0 {
		return result, errors.New("safetensors contains no tensors")
	}
	if role == "text_encoder" {
		patch, merger := false, false
		for name := range tensors {
			if strings.Contains(name, "visual.") {
				patch = patch || strings.Contains(name, "patch_embed")
				merger = merger || strings.Contains(name, "merger.linear_fc2") || strings.Contains(name, "merger.mlp.2")
			}
		}
		result.SupportsEditing = patch && merger
	}
	formats := map[string]bool{}
	for name, tensor := range tensors {
		switch tensor.DType {
		case "F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I32", "I64", "U8":
		case "I8":
		default:
			result.Hint = "Unsupported safetensors dtype: " + tensor.DType
			return result, nil
		}
		if tensor.DType == "F8_E4M3" || tensor.DType == "F8_E5M2" {
			formats["FP8"] = true
		}
		if tensor.DType != "I8" {
			continue
		}
		base := strings.TrimSuffix(name, ".weight")
		meta, ok := tensors[base+".comfy_quant"]
		if !ok || meta.DType != "U8" || meta.Offsets[1]-meta.Offsets[0] > 65536 || meta.Offsets[1] == meta.Offsets[0] {
			result.Hint = "I8 weights require recognized Comfy quantization metadata"
			return result, nil
		}
		data := make([]byte, int(meta.Offsets[1]-meta.Offsets[0]))
		if _, err := f.ReadAt(data, 8+int64(size)+meta.Offsets[0]); err != nil {
			return result, err
		}
		var quant struct {
			Format  string `json:"format"`
			ConvRot bool   `json:"convrot"`
			Group   int64  `json:"convrot_groupsize"`
		}
		if json.Unmarshal(data, &quant) != nil || quant.Format != "int8_tensorwise" {
			result.Hint = "Unsupported packed I8 quantization format"
			return result, nil
		}
		if quant.ConvRot {
			g := quant.Group
			for g > 1 && g%4 == 0 {
				g /= 4
			}
			if g != 1 || quant.Group <= 0 || len(tensor.Shape) < 2 || tensor.Shape[len(tensor.Shape)-1]%quant.Group != 0 {
				result.Hint = "Unsupported ConvRot group shape"
				return result, nil
			}
			formats["INT8 ConvRot"] = true
		} else {
			formats["INT8 tensorwise"] = true
		}
		scale, ok := tensors[base+".weight_scale"]
		if !ok || scale.DType != "F32" {
			result.Hint = "Packed I8 weights require F32 weight_scale"
			return result, nil
		}
	}
	if role == "lora" {
		pairs := 0
		for name, a := range tensors {
			var bName string
			switch {
			case strings.Contains(name, ".lora_A.default.weight"):
				bName = strings.Replace(name, ".lora_A.default.weight", ".lora_B.default.weight", 1)
			case strings.Contains(name, ".lora_A.weight"):
				bName = strings.Replace(name, ".lora_A.weight", ".lora_B.weight", 1)
			case strings.Contains(name, ".lora_down.weight"):
				bName = strings.Replace(name, ".lora_down.weight", ".lora_up.weight", 1)
			default:
				continue
			}
			b, ok := tensors[bName]
			if !ok || len(a.Shape) != 2 || len(b.Shape) != 2 || a.Shape[0] <= 0 || a.Shape[0] != b.Shape[1] {
				result.Hint = "LoRA factor pair is missing or has incompatible rank"
				return result, nil
			}
			pairs++
		}
		if pairs == 0 {
			result.Compatibility = "conversion_required"
			result.Hint = "No verified PEFT/Kohya LoRA factor names found"
			return result, nil
		}
		result.Format = "safetensors LoRA"
		result.Compatibility = "supported"
		result.Hint = "PEFT/Kohya rank pairs recognized; applied at runtime without merging the quantized base"
		return result, nil
	}
	result.Format = "safetensors"
	for _, name := range []string{"FP8", "INT8 ConvRot", "INT8 tensorwise"} {
		if formats[name] {
			result.Format += " · " + name
		}
	}
	result.Compatibility = "supported"
	result.Hint = "Storage and quantization metadata supported by pinned image runner; no weight conversion required"
	return result, nil
}
