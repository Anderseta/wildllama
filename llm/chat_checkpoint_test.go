package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
)

func TestChatCheckpointRestartIsolationAndDeletion(t *testing.T) {
	root := t.TempDir()
	cfg := &imageHandoffConfig{directory: filepath.Join(root, ".image-handoff"), auth: "private", budget: 64}
	if err := os.Mkdir(cfg.directory, 0700); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(root, "runner")
	if err := os.WriteFile(exe, []byte("runner"), 0600); err != nil {
		t.Fatal(err)
	}
	var restores, saves int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var data map[string]any
		if json.NewDecoder(r.Body).Decode(&data) != nil || data["handoff_auth"] != "private" || data["filename"] != handoffFile {
			t.Error("invalid native request")
			w.WriteHeader(400)
			return
		}
		switch r.URL.Query().Get("action") {
		case "save":
			saves++
			if err := os.WriteFile(filepath.Join(cfg.directory, handoffFile), []byte("draft+kv"), 0600); err != nil {
				t.Error(err)
			}
			fmt.Fprint(w, `{"n_written":8}`)
		case "restore":
			restores++
			b, _ := os.ReadFile(filepath.Join(cfg.directory, handoffFile))
			if string(b) != "draft+kv" {
				t.Error("invalid restore")
			}
			fmt.Fprint(w, `{"n_read":8}`)
		}
	}))
	defer server.Close()
	port := server.Listener.Addr().(*net.TCPAddr).Port
	runner := func(model string) *llamaServerRunner {
		return &llamaServerRunner{cmd: exec.Command(exe, "--port", strconv.Itoa(port)), port: port, launch: llamaServerLaunchConfig{handoff: cfg, config: LlamaServerConfig{ManifestDigest: model}}}
	}
	ctx := context.Background()
	r := runner("a")
	c, err := r.beginChatCheckpoint(ctx, "project")
	if err != nil || c == nil {
		t.Fatal(err)
	}
	r.finishChatCheckpoint(ctx, c, 100)
	if saves != 0 {
		t.Fatal("short chat persisted")
	}
	r.finishChatCheckpoint(ctx, c, 6000)
	r.finishChatCheckpoint(ctx, c, 6005)
	if saves != 1 {
		t.Fatal("repeated checkpoint")
	}
	begin := func(r *llamaServerRunner, id string) *chatCheckpoint {
		cc, e := r.beginChatCheckpoint(ctx, id)
		if e != nil {
			t.Fatal(e)
		}
		return cc
	}
	begin(r, "project")
	begin(runner("b"), "project")
	begin(runner("a"), "other")
	if restores != 0 {
		t.Fatal("incompatible or already warm restore")
	}
	r = runner("a")
	begin(r, "project")
	if restores != 1 {
		t.Fatal("missing restart restore")
	}
	files, _, _ := c.files()
	if len(files) != 1 {
		t.Fatal(files)
	}
	if err := os.WriteFile(filepath.Join(c.directory, files[0].name), []byte("corrupt!"), 0600); err != nil {
		t.Fatal(err)
	}
	begin(runner("a"), "project")
	if restores != 1 {
		t.Fatal("corrupt snapshot submitted")
	}
	if err := os.WriteFile(filepath.Join(c.directory, c.owner+".deleted"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	if begin(r, "project") != nil {
		t.Fatal("revoked session admitted")
	}
	r.finishChatCheckpoint(ctx, c, 12000)
	if saves != 1 {
		t.Fatal("deleted checkpoint republished")
	}
}
