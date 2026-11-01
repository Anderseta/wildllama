package server

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestDiscardDownloadUsesLocalMetadataAndCleansVerificationState(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := model.ParseName("huggingface.co/owner/pending:q4")
	digest := "sha256:" + strings.Repeat("a", 64)
	if err := manifest.PinDownload(name, []string{digest}); err != nil {
		t.Fatal(err)
	}
	blob, _ := manifest.BlobsPath(digest)
	if err := os.WriteFile(blob+"-partial", []byte("partial bytes"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := markBlobVerificationPending(digest); err != nil {
		t.Fatal(err)
	}
	w := createRequest(t, (&Server{}).DiscardDownloadHandler, api.DeleteRequest{Model: "hf.co/owner/pending:q4"})
	var result manifest.DownloadCleanup
	if w.Code != http.StatusOK || json.Unmarshal(w.Body.Bytes(), &result) != nil || result.RemovedFiles != 1 {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	if pending, err := blobVerificationPending(digest); err != nil || pending {
		t.Fatalf("left verification metadata: %v %v", pending, err)
	}
	if w := createRequest(t, (&Server{}).DiscardDownloadHandler, api.DeleteRequest{Model: "../../bad"}); w.Code != http.StatusBadRequest {
		t.Fatalf("invalid name status %d", w.Code)
	}
}

func TestDiscardDownloadRefusesConcurrentPullOrImport(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	downloadCleanupGate.RLock()
	defer downloadCleanupGate.RUnlock()
	w := createRequest(t, (&Server{}).DiscardDownloadHandler, api.DeleteRequest{Model: "pending"})
	if w.Code != http.StatusConflict {
		t.Fatalf("unsafe cleanup during active download: %d %s", w.Code, w.Body.String())
	}
}
