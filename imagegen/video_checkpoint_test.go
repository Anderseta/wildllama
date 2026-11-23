//go:build windows || linux

package imagegen

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestWanCheckpointRoundTrip(t *testing.T) {
	for _, step := range []int{1, 2} {
		const size = 1024
		m := newWanReference(16 + 2*size)
		if m.view == nil {
			t.Fatal("shared mapping unavailable")
		}
		c := &VideoCheckpoint{memory: m, size: 16 + 2*size, step: step, seed: 98765432109876}
		p := filepath.Join(t.TempDir(), "steps", "video.wanstep")
		payload := bytes.Repeat([]byte{byte(step)}, size)
		if err := m.view(int(c.size), func(b []byte) error {
			binary.LittleEndian.PutUint32(b, uint32(step))
			binary.LittleEndian.PutUint32(b[4:], 427)
			copy(b[16+(step%2)*size:], payload)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		if err := c.Save(p); err != nil {
			t.Fatal(err)
		}
		c.Close()
		saved, err := SavedVideoStep(p, size, 4)
		if err != nil || saved != step {
			t.Fatalf("saved step %d: %v", saved, err)
		}
		loaded, err := LoadVideoCheckpoint(p, size, 4)
		if err != nil {
			t.Fatal(err)
		}
		if loaded.seed != 98765432109876 || loaded.memory.publishedStep(4) != step {
			t.Fatal("lost seed or step")
		}
		err = loaded.memory.view(int(loaded.size), func(b []byte) error {
			offset := 16 + (step%2)*size
			if binary.LittleEndian.Uint32(b[4:]) != 427 || !bytes.Equal(b[offset:offset+size], payload) {
				t.Fatal("lost schedule or latent")
			}
			return nil
		})
		loaded.Close()
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != 32+size {
			t.Fatalf("extra slot persisted: %d bytes", len(data))
		}
		data[len(data)-1] ^= 127
		if err = os.WriteFile(p, data, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err = LoadVideoCheckpoint(p, size, 4); err == nil {
			t.Fatal("damaged payload accepted")
		}
	}
}
