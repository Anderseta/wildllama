package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCheckpointSelectsEditedBranchBeforeReadingState(t *testing.T) {
	c := chatCheckpoint{directory: t.TempDir(), owner: strings.Repeat("1", 64), runtime: strings.Repeat("2", 64), maxFile: 65536}
	encode := func(tokens []int) []byte {
		var data []byte
		for _, v := range []uint32{0x67677371, 4, uint32(len(tokens) + 4), 0xffffffff, 1, uint32(len(tokens))} {
			data = binary.LittleEndian.AppendUint32(data, v)
		}
		for _, v := range tokens {
			data = binary.LittleEndian.AppendUint32(data, uint32(v))
		}
		return binary.LittleEndian.AppendUint32(data, 0)
	}
	save := func(b []byte) checkpointFile {
		name := fmt.Sprintf("%s%x.slot", c.prefix(), sha256.Sum256(b))
		if err := os.WriteFile(filepath.Join(c.directory, name), b, 0600); err != nil {
			t.Fatal(err)
		}
		return checkpointFile{name: name, size: int64(len(b))}
	}
	first := make([]int, 600)
	for i := range first {
		first[i] = i % 70
	}
	old := save(encode(first))
	changed := append([]int{}, first...)
	changed[330] = 99
	raw := encode(changed)
	delta := append([]byte(deltaMagic+old.name[130:194]), 1)
	delta = binary.LittleEndian.AppendUint32(delta, uint32(len(raw)))
	recent := save(append(append(delta, raw...), 0))
	for _, test := range []struct {
		prompt []int
		want   string
	}{{first, old.name}, {changed, recent.name}, {[]int{99, 88}, ""}} {
		candidates := c.restoreCandidates([]checkpointFile{recent, old}, func() ([]int, error) { return test.prompt, nil })
		if test.want == "" {
			if len(candidates) != 0 {
				t.Fatal("restoring unrelated history")
			}
		} else if len(candidates) == 0 || candidates[0].name != test.want {
			t.Fatalf("wrong branch: %v", candidates)
		}
	}
	other := c
	other.owner = strings.Repeat("3", 64)
	if files := other.restoreCandidates([]checkpointFile{old}, func() ([]int, error) { t.Fatal("tokenized unrelated chat"); return nil, nil }); len(files) != 0 {
		t.Fatal(files)
	}
	matched, full, err := checkpointMatch(bytes.NewReader(encode(first)[:100]), first)
	if err == nil || full || matched != 0 {
		t.Fatal("accepted truncated token list")
	}
}

func TestCheckpointUsesPreparedWireTokens(t *testing.T) {
	// A prepared ChatRequest no longer owns its messages, tool schema or template parameters.
	wire := []byte(`{"messages":[{"role":"user","content":"edited"}],"tools":[{"type":"function"}],"count_tokens":true}`)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil || !bytes.Equal(body, wire) || r.URL.Path != "/apply-template" ||
			r.Header.Get("X-WildLlama-Checkpoint-Tokens") != "1" || r.Header.Get(preparedChatHeader) != "" {
			t.Errorf("incorrect checkpoint template request: %s", body)
		}
		fmt.Fprint(w, `{"tokens":[1,2,3]}`)
	}))
	defer server.Close()
	runner := llamaServerRunner{port: server.Listener.Addr().(*net.TCPAddr).Port}
	tokens, err := runner.checkpointPromptTokens(context.Background(), wire)
	if err != nil || len(tokens) != 3 || tokens[2] != 3 {
		t.Fatal(tokens, err)
	}
}
