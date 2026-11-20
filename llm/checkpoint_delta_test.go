package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func copyRecord(n uint32, offset uint64) []byte {
	data := []byte{2}
	data = binary.LittleEndian.AppendUint32(data, n)
	return binary.LittleEndian.AppendUint64(data, offset)
}

func TestCheckpointDeltaRecords(t *testing.T) {
	base := []byte("0123456789abcdef")
	records := append(copyRecord(8, 4), 1, 2, 0, 0, 0, 'x', 'y', 0)
	var out bytes.Buffer
	n, err := expandCheckpoint(bytes.NewReader(records), bytes.NewReader(base), 16, &out, 10)
	if err != nil || n != 10 || out.String() != "456789abxy" {
		t.Fatal(out.String(), n, err)
	}
	for _, bad := range [][]byte{records[:len(records)-1], append(bytes.Clone(records), 0), copyRecord(2, 15),
		copyRecord(1, ^uint64(0)), {1, 2, 0, 0, 0, 5}, {1, 0, 0, 0, 0}, {9}, {0}} {
		if _, err := expandCheckpoint(bytes.NewReader(bad), bytes.NewReader(base), 16, io.Discard, 10); err == nil {
			t.Fatalf("accepted corrupt delta %x", bad)
		}
	}
	if _, err := expandCheckpoint(bytes.NewReader(records), bytes.NewReader(base), 16, io.Discard, 9); err == nil {
		t.Fatal("expanded beyond budget")
	}
}

func TestCheckpointDeltaIsolation(t *testing.T) {
	dir := t.TempDir()
	c := chatCheckpoint{directory: dir, owner: strings.Repeat("a", 64), runtime: strings.Repeat("b", 64), maxFile: 1024}
	base := []byte(strings.Repeat("shared state ", 20))
	digest := fmt.Sprintf("%x", sha256.Sum256(base))
	name := c.prefix() + digest + ".slot"
	if err := os.WriteFile(filepath.Join(dir, name), base, 0600); err != nil {
		t.Fatal(err)
	}
	data := append([]byte(deltaMagic+digest), copyRecord(40, 6)...)
	data = append(data, 0)
	src := filepath.Join(dir, c.prefix()+fmt.Sprintf("%x.slot", sha256.Sum256(data)))
	if err := os.WriteFile(src, data, 0600); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(dir, "restore")
	n, err := c.materialize(src, dst, int64(len(data)), 1024)
	got, _ := os.ReadFile(dst)
	if err != nil || n != 40 || !bytes.Equal(got, base[6:46]) {
		t.Fatal(n, err, string(got))
	}
	other := c
	other.owner = strings.Repeat("c", 64)
	if _, err := other.materialize(src, dst+"2", int64(len(data)), 1024); err == nil {
		t.Fatal("cross-session base")
	}
	if err = os.WriteFile(filepath.Join(dir, name), bytes.Repeat([]byte("z"), len(base)), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := c.materialize(src, dst+"3", int64(len(data)), 1024); err == nil {
		t.Fatal("accepted modified keyframe")
	}
}

func TestChatCheckpointIncrementalPublishing(t *testing.T) {
	t.Setenv(checkpointDeltaEnv, "1")
	root := t.TempDir()
	cfg := &imageHandoffConfig{directory: filepath.Join(root, ".image-handoff"), auth: "private", budget: 4096}
	if err := os.Mkdir(cfg.directory, 0700); err != nil {
		t.Fatal(err)
	}
	base := bytes.Repeat([]byte("abcdef01"), 100)
	digest := fmt.Sprintf("%x", sha256.Sum256(base))
	next := append(bytes.Clone(base), 42)
	scratch := filepath.Join(cfg.directory, handoffFile)
	restored := false
	saves := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			w.WriteHeader(400)
			return
		}
		if r.URL.Query().Get("action") == "restore" {
			b, _ := os.ReadFile(scratch)
			if body["checkpoint_base"] != nil {
				pinned, err := os.ReadFile(scratch + ".base")
				restored = err == nil && bytes.Equal(pinned, base) && body["checkpoint_base"] == digest &&
					bytes.HasPrefix(b, []byte(deltaMagic+digest))
			} else {
				restored = bytes.Equal(next, b)
			}
			fmt.Fprintf(w, "{\"n_read\":%d}", len(next))
			return
		}
		data := base
		if saves > 0 {
			input, _ := os.ReadFile(scratch + ".base")
			if !bytes.Equal(input, base) || body["checkpoint_base"] != digest {
				t.Error("wrong base")
			}
			data = append([]byte(deltaMagic+digest), copyRecord(uint32(len(base)), 0)...)
			data = append(data, 1, 1, 0, 0, 0, 42, 0)
		}
		saves++
		if err := os.WriteFile(scratch, data, 0600); err != nil {
			t.Error(err)
		}
		fmt.Fprintf(w, "{\"n_written\":%d}", len(data))
	}))
	defer srv.Close()
	port := srv.Listener.Addr().(*net.TCPAddr).Port
	exe, _ := os.Executable()
	runner := func() *llamaServerRunner {
		return &llamaServerRunner{cmd: exec.Command(exe), port: port,
			launch: llamaServerLaunchConfig{handoff: cfg, config: LlamaServerConfig{ManifestDigest: "local"}}}
	}
	r := runner()
	ctx := context.Background()
	c, err := r.beginChatCheckpoint(ctx, "incremental", nil)
	if err != nil || c == nil {
		t.Fatal(err)
	}
	r.finishChatCheckpoint(ctx, c, 2000, nil)
	r.finishChatCheckpoint(ctx, c, 5000, nil)
	files, _, err := c.files()
	if err != nil || len(files) != 2 || saves != 2 {
		t.Fatal(files, saves, err)
	}
	// A profile may opt out again between saving and restarting.
	t.Setenv(checkpointDeltaEnv, "0")
	if _, err = runner().beginChatCheckpoint(ctx, "incremental", nil); err != nil || !restored {
		t.Fatal(err, restored)
	}
	cfg.directRestore = true
	restored = false
	if _, err = runner().beginChatCheckpoint(ctx, "incremental", nil); err != nil || !restored {
		t.Fatal("native restore did not receive both immutable inputs", err)
	}
	if _, err = os.Stat(scratch + ".base"); !os.IsNotExist(err) {
		t.Fatal("baseline link leaked", err)
	}
}

func TestCheckpointQuotaRecoversAfterReduction(t *testing.T) {
	c := chatCheckpoint{directory: t.TempDir(), owner: strings.Repeat("a", 64), runtime: strings.Repeat("b", 64), maxFile: 100, budget: 300}
	write := func(size int) string {
		data := bytes.Repeat([]byte("x"), size)
		name := fmt.Sprintf("%s%x.slot", c.prefix(), sha256.Sum256(data))
		if err := os.WriteFile(filepath.Join(c.directory, name), data, 0600); err != nil {
			t.Fatal(err)
		}
		return name
	}
	old := write(280)
	good := write(80)
	list, used, err := c.files()
	if err != nil || !c.reserveSaveSpace(list, used) {
		t.Fatal("reservation failed", err)
	}
	if _, err := os.Stat(filepath.Join(c.directory, old)); !os.IsNotExist(err) {
		t.Fatal("oversized slot retained")
	}
	if _, err := os.Stat(filepath.Join(c.directory, good)); err != nil {
		t.Fatal("lost usable slot", err)
	}
}
