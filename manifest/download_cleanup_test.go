package manifest

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestDiscardDownloadKeepsInstalledAndOtherPendingWeights(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	layer := func(value string) Layer {
		t.Helper()
		l, err := NewLayer(strings.NewReader(value), MediaTypeImageModel)
		if err != nil {
			t.Fatal(err)
		}
		return l
	}
	weights, projector, shared, unrelated := layer("weights"), layer("projector"), layer("shared"), layer("unrelated")
	config, err := NewLayer(strings.NewReader(`{}`), MediaTypeImageConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err := WriteManifest(model.ParseName("installed"), config, []Layer{shared}); err != nil {
		t.Fatal(err)
	}
	name := model.ParseName("huggingface.co/owner/model:q4")
	if err := PinDownload(name, []string{weights.Digest, projector.Digest, shared.Digest}); err != nil {
		t.Fatal(err)
	}
	if err := PinDownload(model.ParseName("another-pending"), []string{"sha256:" + strings.ToUpper(projector.Digest[7:])}); err != nil {
		t.Fatal(err)
	}
	weightPath, _ := BlobsPath(weights.Digest)
	var deleted []string
	for _, suffix := range []string{"", "-partial", "-partial-0", "-partial-15"} {
		file := weightPath + suffix
		if err := os.WriteFile(file, []byte("partial"), 0644); err != nil {
			t.Fatal(err)
		}
		deleted = append(deleted, file)
	}
	odd := weightPath + "-partial-not-a-number"
	if err := os.WriteFile(odd, []byte("unrelated"), 0644); err != nil {
		t.Fatal(err)
	}
	result, err := DiscardDownload(name)
	if err != nil || result.RemovedFiles != 4 || result.RetainedBlobs != 2 {
		t.Fatalf("cleanup: %+v, %v", result, err)
	}
	for _, file := range deleted {
		if _, err := os.Stat(file); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("retained canceled artifact %s: %v", file, err)
		}
	}
	for _, digest := range []string{projector.Digest, shared.Digest, unrelated.Digest} {
		path, _ := BlobsPath(digest)
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("lost retained weights: %s: %v", path, err)
		}
	}
	if _, err := os.Stat(odd); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(downloadPinPath(name)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("pin remains: %v", err)
	}
	if again, err := DiscardDownload(name); err != nil || again.RemovedFiles != 0 {
		t.Fatalf("non-idempotent cleanup: %+v %v", again, err)
	}
}

func TestDiscardDownloadFailsClosedOnCorruptMetadataAndInstalledManifest(t *testing.T) {
	for _, corrupt := range []string{"pin", "manifest", "directory"} {
		t.Run(corrupt, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			weights, err := NewLayer(strings.NewReader("keep on error"), MediaTypeImageModel)
			if err != nil {
				t.Fatal(err)
			}
			name := model.ParseName("pending")
			if err := PinDownload(name, []string{weights.Digest}); err != nil {
				t.Fatal(err)
			}
			switch corrupt {
			case "pin":
				if err := os.WriteFile(downloadPinPath(name), []byte(`{"model":"pending","digests":["../../outside"]}`), 0644); err != nil {
					t.Fatal(err)
				}
			case "manifest":
				path, _ := V2PathForName(model.ParseName("broken"))
				if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte("{"), 0644); err != nil {
					t.Fatal(err)
				}
			case "directory":
				path, _ := BlobsPath(weights.Digest)
				if err := os.Mkdir(path+"-partial", 0755); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := DiscardDownload(name); err == nil {
				t.Fatal("unsafe cleanup accepted")
			}
			path, _ := BlobsPath(weights.Digest)
			if _, err := os.Stat(path); err != nil {
				t.Fatal("deleted before validating all references")
			}
			if _, err := os.Stat(downloadPinPath(name)); err != nil {
				t.Fatal("lost retry metadata")
			}
		})
	}
}
