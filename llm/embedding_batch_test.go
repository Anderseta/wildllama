package llm

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestNativeEmbeddingBatchOrderAndTokenization(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if r.URL.Path == "/tokenize" {
			fmt.Fprint(w, `{"tokens_batch":[[1,2],[3]]}`)
			return
		}
		if body["encoding_format"] != "base64" || len(body["input"].([]any)) != 2 {
			t.Error("not a real array request")
		}
		fmt.Fprint(w, `{"data":[{"index":1,"embedding":[3,4]},{"index":0,"embedding":[1,2]}],"usage":{"prompt_tokens":5}}`)
	}))
	defer server.Close()
	address, _ := url.Parse(server.URL)
	port, _ := strconv.Atoi(address.Port())
	runner := &llamaServerRunner{port: port, sem: semaphore.NewWeighted(4), launch: llamaServerLaunchConfig{embedding: true, numParallel: 4, opts: api.Options{Runner: api.Runner{NumBatch: 512}}}}
	tokens, err := runner.TokenizeBatch(t.Context(), []string{"a", "b"})
	if err != nil || len(tokens) != 2 || len(tokens[0]) != 2 {
		t.Fatalf("tokenization: %v", err)
	}
	vectors, count, err := runner.EmbeddingBatch(t.Context(), []string{"a", "b"})
	if err != nil || count != 5 || vectors[0][0] != 1 || vectors[1][0] != 3 {
		t.Fatalf("embedding ordering or usage: %v", err)
	}
}

func TestPackedNativeEmbedding(t *testing.T) {
	bits := []uint32{0, 1, 0x80000000, 0x3eaaaaab, 0xc0000000, 0x7f7fffff}
	raw := make([]byte, 4*len(bits))
	for i, b := range bits {
		binary.NativeEndian.PutUint32(raw[i*4:], b)
	}
	encoded, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(fmt.Sprintf(`{"data":[{"embedding":%s,"tokens_evaluated":7}]}`, encoded))
	values, count, err := decodeEmbeddingResponse(body)
	if err != nil || count != 7 || len(values) != len(bits) {
		t.Fatalf("decode: %v, %d, %v", values, count, err)
	}
	for i, v := range values {
		if math.Float32bits(v) != bits[i] {
			t.Fatalf("element %d changed bits", i)
		}
	}
	for _, invalid := range []string{`"!"`, `"AA=="`, `""`} {
		body := []byte(fmt.Sprintf(`{"data":[{"embedding":%s}]}`, invalid))
		if _, _, err := decodeEmbeddingResponse(body); err == nil {
			t.Fatalf("accepted invalid packed vector %s", invalid)
		}
	}
	binary.NativeEndian.PutUint32(raw, 0x7fc00000)
	encoded, _ = json.Marshal(raw)
	var vector nativeEmbeddingVector
	if err := json.Unmarshal(encoded, &vector); err == nil {
		t.Fatal("accepted non-finite embedding")
	}
}

func TestDecodeSingleEmbeddingRejectsIncompleteResults(t *testing.T) {
	for _, body := range []string{`{"data":[{"embedding":null}]}`, `{"data":[{"embedding":[]}]}`, `[]`, `[{"embedding":[[]]}]`, `{"data":[{"embedding":[1]},{"embedding":[2]}]}`} {
		if _, _, err := decodeEmbeddingResponse([]byte(body)); err == nil {
			t.Errorf("accepted %s", body)
		}
	}
}
