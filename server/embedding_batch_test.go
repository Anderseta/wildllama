package server

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

type batchFixture struct {
	mockLlm
	groups [][]string
	fail   bool
}

func (f *batchFixture) EmbeddingBatchLimits() (int, int) { return 3, 12 }
func (f *batchFixture) TokenizeBatch(ctx context.Context, inputs []string) ([][]int, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := make([][]int, len(inputs))
	for i, value := range inputs {
		result[i] = make([]int, len(strings.Fields(value)))
	}
	return result, nil
}
func (f *batchFixture) EmbeddingBatch(ctx context.Context, inputs []string) ([][]float32, int, error) {
	f.groups = append(f.groups, inputs)
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if f.fail {
		return nil, 0, api.StatusError{StatusCode: 413, ErrorMessage: "too large"}
	}
	result := make([][]float32, len(inputs))
	for i, value := range inputs {
		result[i] = []float32{float32(len(value)), 1}
	}
	return result, len(inputs), nil
}
func TestEmbeddingBatchGroupingDedupAndFallback(t *testing.T) {
	for _, fail := range []bool{false, true} {
		fixture := &batchFixture{fail: fail}
		cache := map[string][]int{}
		singles := 0
		inputs := []embeddingInput{{text: "one two"}, {text: "short"}, {text: "one two"}, {text: "three four five"}}
		result, _, used, err := prepareEmbeddingBatch(t.Context(), fixture, inputs,
			func(text string, tokens []int) { cache[text] = tokens },
			func() { clear(cache) },
			func(text string) ([]int, error) { return cache[text], nil },
			func(text string) (string, bool, error) { return text, false, nil },
			func(item embeddingInput) ([]float32, int, error) {
				singles++
				return []float32{float32(len(item.text)), 1}, 1, nil
			}, rawEmbedding)
		if err != nil || !used || len(result) != len(inputs) {
			t.Fatalf("batch result: %v %v", used, err)
		}
		if len(fixture.groups) != 1 {
			t.Fatalf("expected one native batch, got %d", len(fixture.groups))
		}
		if fail && singles != 3 {
			t.Fatalf("fallback should process only unique inputs: %d", singles)
		}
		for i, item := range inputs {
			if result[i][0] != float32(len(item.text)) {
				t.Fatal("order changed")
			}
		}
		result[0][0] = -1
		if result[2][0] < 0 {
			t.Fatal("duplicate outputs alias each other")
		}
	}
}

func rawEmbedding(v []float32) ([]float32, error) { return v, nil }

func TestFinalizeEmbeddingBatchOncePerPreparedText(t *testing.T) {
	f := &embeddingRetentionFixture{cache: make(map[string][]int)}
	input := []embeddingInput{{text: "long"}, {text: "other"}, {text: "trimmed"}, {text: "long"}}
	prepare := func(s string) (string, bool, error) {
		if s == "trimmed" {
			return "long", true, nil
		}
		return s, false, nil
	}
	calls := 0
	finish := func(vector []float32) ([]float32, error) {
		calls++
		return finalizeEmbedding(vector, 128, []int{128, 768})
	}
	got, count, used, err := prepareEmbeddingBatch(t.Context(), f, input, f.remember, f.clear, f.tokens, prepare, f.single, finish)
	if err != nil || !used || calls != 2 || count != 2048 || len(got) != 4 {
		t.Fatalf("count=%d calls=%d used=%v err=%v", count, calls, used, err)
	}
	for _, v := range got {
		if len(v) != 128 || v[0] != 1 {
			t.Fatalf("unexpected vector of length %d", len(v))
		}
	}
	got[0][0] = 0
	if got[2][0] != 1 || got[3][0] != 1 {
		t.Fatal("repeated outputs share storage")
	}
}
func TestFinalizeEmbeddingPreservesNormalization(t *testing.T) {
	input := []float32{.25, -8.3, 1.8, 4.2}
	first, _ := normalize(slices.Clone(input))
	want, _ := normalize(first[:2])
	got, err := finalizeEmbedding(slices.Clone(input), 2, []int{2, 4})
	if err != nil || !slices.Equal(got, want) {
		t.Fatal(got, want, err)
	}
	for _, dim := range []int{3, 5} {
		if _, err := finalizeEmbedding(slices.Clone(input), dim, []int{2, 4}); err == nil {
			t.Fatal("accepted dimension", dim)
		}
	}
	if _, err := finalizeEmbedding([]float32{1, 2, float32(math.Inf(1))}, 2, nil); err == nil {
		t.Fatal("accepted nonfinite suffix")
	}
	f := &embeddingRetentionFixture{cache: make(map[string][]int)}
	fail := errors.New("invalid output")
	_, _, _, err = prepareEmbeddingBatch(t.Context(), f, []embeddingInput{{text: "a"}, {text: "a"}}, f.remember, f.clear, f.tokens, f.prepare, f.single, func([]float32) ([]float32, error) { return nil, fail })
	if !errors.Is(err, fail) {
		t.Fatal(err)
	}
}
