package manifest

import (
	"errors"
	"os"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestLocalizeModelNamesPreservesManifestAndOrigin(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	source := model.ParseName("huggingface.co/Author/Model:Q8")
	data := []byte(`{"schemaVersion":2}`)
	if err := WriteManifestData(source, data); err != nil {
		t.Fatal(err)
	}
	for attempt, want := range []int{1, 0} {
		count, err := LocalizeModelNames()
		if err != nil || count != want {
			t.Fatalf("attempt %d: %d %v", attempt, count, err)
		}
	}
	if actual, err := ReadManifestData(source.LocalName()); err != nil || string(actual) != string(data) {
		t.Fatalf("manifest lost: %s %v", actual, err)
	}
	if _, err := ReadManifestData(source); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("old name retained: %v", err)
	}
	if origin := ModelOrigin(source.LocalName()); origin != source.DisplayShortest() {
		t.Fatalf("origin lost: %s", origin)
	}
}

func TestLocalizeModelNamesRejectsConflictingContent(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	source := model.ParseName("huggingface.co/Author/Model:Q8")
	for name, data := range map[model.Name]string{source: `{"schemaVersion":2}`, source.LocalName(): `{"schemaVersion":1}`} {
		if err := WriteManifestData(name, []byte(data)); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := LocalizeModelNames(); err == nil {
		t.Fatal("overwrote conflicting local model")
	}
	for name := range map[model.Name]bool{source: true, source.LocalName(): true} {
		if _, err := ReadManifestData(name); err != nil {
			t.Fatal(err)
		}
	}
}
