package llm

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func TestImageHandoffLifecycle(t *testing.T) {
	for _, test := range []string{"restore", "corrupt", "incompatible", "failed-clear"} {
		t.Run(test, func(t *testing.T) {
			dir := t.TempDir()
			exe := filepath.Join(dir, "runner")
			if err := os.WriteFile(exe, []byte("runtime identity"), 0600); err != nil {
				t.Fatal(err)
			}
			cfg := &imageHandoffConfig{auth: "private", directory: dir, budget: 64}
			path := filepath.Join(dir, handoffFile)
			var restores, erases int
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Auth   string `json:"handoff_auth"`
					Budget int    `json:"image_budget"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Auth != "private" || body.Budget != 64 {
					http.Error(w, "bad request", 400)
					return
				}
				switch r.URL.Query().Get("action") {
				case "save":
					if err := os.WriteFile(path, []byte("prefix"), 0600); err != nil {
						t.Error(err)
					}
					w.Write([]byte(`{"n_written":6}`))
				case "restore":
					restores++
					if test == "failed-clear" {
						http.Error(w, "damaged", 400)
						return
					}
					w.Write([]byte(`{"n_read":6}`))
				case "erase":
					erases++
					http.Error(w, "unavailable", 503)
				}
			}))
			defer srv.Close()
			port := srv.Listener.Addr().(*net.TCPAddr).Port
			runner := &llamaServerRunner{cmd: exec.Command(exe, "--port", strconv.Itoa(port)), port: port, launch: llamaServerLaunchConfig{handoff: cfg, config: LlamaServerConfig{ManifestDigest: "a"}}}
			runner.SaveImageHandoff(context.Background())
			if _, err := os.Stat(path); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				suspendedImagePrefix.Lock()
				defer suspendedImagePrefix.Unlock()
				if r := suspendedImagePrefix.receipt; r != nil {
					r.discard()
					suspendedImagePrefix.receipt = nil
				}
			})
			if lock, err := cfg.lock(); err == nil {
				lock.Close()
				t.Fatal("parallel writer acquired handoff lease")
			}
			if test == "corrupt" {
				if err := os.WriteFile(path, []byte("PREFIX"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if test == "incompatible" {
				runner.launch.config.ManifestDigest = "b"
				if err := runner.restoreImageHandoff(context.Background()); err != nil || restores != 0 {
					t.Fatalf("incompatible restore: %v", err)
				}
				runner.launch.config.ManifestDigest = "a"
			}
			err := runner.restoreImageHandoff(context.Background())
			if (err != nil) != (test == "failed-clear") {
				t.Fatalf("restore: %v", err)
			}
			if (restores == 1) != (test != "corrupt") || (erases == 1) != (test == "failed-clear") {
				t.Fatalf("restore=%d erase=%d", restores, erases)
			}
			if _, err := os.Stat(path); !os.IsNotExist(err) {
				t.Fatal("checkpoint not discarded")
			}
		})
	}
}

func TestImageHandoffCanceledSaveReleasesLock(t *testing.T) {
	dir := t.TempDir()
	exe := filepath.Join(dir, "runner")
	if err := os.WriteFile(exe, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	started, release := make(chan struct{}), make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(started)
		<-release
		w.Write([]byte(`{"n_written":0}`))
	}))
	defer srv.Close()
	cfg := &imageHandoffConfig{directory: dir, auth: "test", budget: 64}
	runner := &llamaServerRunner{cmd: exec.Command(exe), port: srv.Listener.Addr().(*net.TCPAddr).Port, launch: llamaServerLaunchConfig{handoff: cfg}}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	returned := make(chan struct{})
	go func() { runner.SaveImageHandoff(ctx); close(returned) }()
	<-started
	cancel()
	select {
	case <-returned:
	case <-time.After(2 * time.Second):
		close(release)
		t.Fatal("cancellation blocked on the native writer")
	}
	if l, err := cfg.lock(); err == nil {
		l.Close()
		close(release)
		t.Fatal("file lock released during native write")
	}
	close(release)
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if l, err := cfg.lock(); err == nil {
			l.Close()
			if _, err := os.Stat(filepath.Join(dir, handoffFile)); !os.IsNotExist(err) {
				t.Fatal("canceled checkpoint still present")
			}
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("file lock retained after native write completed")
}
