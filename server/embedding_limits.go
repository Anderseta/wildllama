package server

import (
	"encoding/base64"
	"fmt"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
)

// Validate aggregate work before decoding media or reserving a model. A compact
// list of repeated strings can otherwise allocate arbitrarily many full vectors.
func validateEmbeddingInput(input any, limits inferenceLimits) (int, error) {
	count := 1
	if batch, ok := input.([]any); ok {
		count = len(batch)
	}
	if input == nil {
		count = 0
	}
	if count > limits.EmbedItems {
		return 0, fmt.Errorf("embedding input exceeds %d items", limits.EmbedItems)
	}
	var decoded int64
	check := func(value any) error {
		if item, ok := value.(map[string]any); ok {
			for _, key := range []string{"image", "audio"} {
				encoded, ok := item[key].(string)
				if !ok {
					continue
				}
				// DecodedLen is an upper bound. Do not allocate to establish it.
				size := int64(base64.StdEncoding.DecodedLen(len(encoded)))
				if strings.HasSuffix(encoded, "==") {
					size -= 2
				} else if strings.HasSuffix(encoded, "=") {
					size--
				}
				if size > 0 {
					decoded += size
				}
				if decoded > limits.DecodedBytes {
					return fmt.Errorf("embedding media exceeds %d decoded bytes", limits.DecodedBytes)
				}
			}
		}
		return nil
	}
	if batch, ok := input.([]any); ok {
		for _, item := range batch {
			if err := check(item); err != nil {
				return 0, err
			}
		}
	} else if err := check(input); err != nil {
		return 0, err
	}
	return count, nil
}

func validateEmbeddingOutput(count int, m *Model, req api.EmbedRequest, limits inferenceLimits) error {
	_, err := embeddingOutputBytes(count, m, limits)
	return err
}

func embeddingOutputBytes(count int, m *Model, limits inferenceLimits) (int64, error) {
	dimensions := int64(m.metadata.Int("embedding_length"))
	if dimensions <= 0 {
		dimensions = int64(m.Config.EmbedLen)
	}
	if dimensions <= 0 {
		dimensions = 65536
	} // Unknown metadata must not disable the allocation guard.
	if count > 0 && dimensions > limits.OutputBytes/4/int64(count) {
		return 0, fmt.Errorf("embedding output exceeds %d bytes; reduce the input batch", limits.OutputBytes)
	}
	return int64(count) * dimensions * 4, nil
}

func shrinkEmbeddingReservation(c *gin.Context, outputBytes int64) error {
	value, exists := c.Get("wildllama_reservation")
	if !exists {
		return nil
	}
	reservation, ok := value.(*inferenceReservation)
	if !ok {
		return nil
	}
	base, ok := c.Get("wildllama_input_reservation")
	if !ok {
		return nil
	}
	input, ok := base.(int64)
	if !ok {
		return nil
	}
	if value, exists := c.Get("wildllama_body_counter"); exists {
		if counter, ok := value.(*inferenceBodyCounter); ok {
			input = counter.read*3 + (1 << 20)
		}
	}
	// Release conservative output capacity before inference. Small embedding
	// batches then use all available slots, without a second acquisition that
	// could deadlock while holding already parsed input buffers.
	if outputBytes < 0 || !reservation.shrink(input+3*outputBytes) {
		return errInferenceSize
	}
	return nil
}
