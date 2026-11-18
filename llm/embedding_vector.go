package llm

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
)

// The local llama-server encodes the bytes of its float32 vector in native
// byte order. Older runners can still return a JSON array of numbers.
type nativeEmbeddingVector []float32

func (v *nativeEmbeddingVector) UnmarshalJSON(data []byte) error {
	if len(data) == 0 || data[0] != '"' {
		return json.Unmarshal(data, (*[]float32)(v))
	}
	var raw []byte
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if len(raw)%4 != 0 {
		return errors.New("invalid byte length for float32 embedding")
	}
	values := make([]float32, len(raw)/4)
	for i := range values {
		bits := binary.NativeEndian.Uint32(raw[4*i:])
		if bits&0x7f800000 == 0x7f800000 {
			return errors.New("non-finite float32 in embedding")
		}
		values[i] = math.Float32frombits(bits)
	}
	*v = values
	return nil
}
