package imagegen

import (
	"testing"
	"time"
)

func TestProgressNativeStagedQwenVAETilesRemainDecoding(t *testing.T) {
	// The pinned Qwen runner announces decoding before loading VAE weights,
	// then prints the same bare N/M bars used by diffusion. The real 512/2
	// execution reported 62/128 loaded tensors followed by tiles 2/9 and 4/9.
	sequence := []struct {
		line, phase    string
		current, total int
	}{
		{"[INFO] using VAE for encoding / decoding", "encoding", 0, 0},
		{"[INFO] sampling using Euler method", "sampling", 0, 0},
		{"[INFO] get_learned_condition completed, taking 3.20s", "encoding", 0, 0},
		{"[INFO] generating image: 1/1 - seed 42", "sampling", 0, 0},
		{"loading 492/526 tensors from diffusion_model.safetensors", "loading", 492, 526},
		{"|======> | 1/2 - 8.00s/it\x1b[K", "sampling", 1, 2},
		{"|========| 2/2 - 8.00s/it\x1b[K", "sampling", 2, 2},
		{"[INFO] decoding 1 latents", "decoding", 0, 0},
		{"loading 62/128 tensors from vae.safetensors", "loading", 62, 128},
		{"|====> | 128/128 - 512.00MiB/s\x1b[K", "loading", 128, 128},
		{"|==>   | 2/9 - 2.00it/s\x1b[K", "decoding", 2, 9},
		{"|====> | 4/9 - 0.50s/it\x1b[K", "decoding", 4, 9},
		{"|======| 9/9 - 2.00it/s\x1b[K", "decoding", 9, 9},
		{"[INFO] generating image: 2/2 - seed 43", "sampling", 0, 0},
		{"|>     | 1/25 - 0.50s/it\x1b[K", "sampling", 1, 25},
	}
	var updates []Progress
	log := progressLog{start: time.Now(), callback: func(update Progress) { updates = append(updates, update) }}
	for _, item := range sequence {
		before := len(updates)
		if _, err := log.Write([]byte(item.line + "\r")); err != nil {
			t.Fatal(err)
		}
		if len(updates) != before+1 {
			t.Fatalf("line %q produced %d updates", item.line, len(updates)-before)
		}
		actual := updates[len(updates)-1]
		if actual.Phase != item.phase || actual.Current != item.current || actual.Total != item.total {
			t.Fatalf("line %q: got %#v, want %s %d/%d", item.line, actual, item.phase, item.current, item.total)
		}
	}
}

func TestProgressNativeSemanticResetAndTransfers(t *testing.T) {
	for _, reset := range []string{"sampling using Euler method", "sampling(high noise) using Euler method", "generating image: 1/1 - seed 1"} {
		t.Run(reset, func(t *testing.T) {
			var last Progress
			log := progressLog{start: time.Now(), callback: func(update Progress) { last = update }}
			log.Write([]byte("decoding VAE tile 1/9\r"))
			log.Write([]byte("loading 42/128 tensors\r"))
			log.Write([]byte("|==> | 2/9 - 0.50s/it\r"))
			if last.Phase != "decoding" {
				t.Fatalf("VAE phase was lost: %#v", last)
			}
			log.Write([]byte(reset + "\r"))
			log.Write([]byte("|==> | 1/25 - 0.50s/it\r"))
			if last.Phase != "sampling" || last.Total != 25 {
				t.Fatalf("sampling did not restart: %#v", last)
			}
			log.Write([]byte("encoding reference image\r"))
			log.Write([]byte("|==> | 2/9 - 0.50s/it\r"))
			if last.Phase != "encoding" {
				t.Fatalf("reference tiles claimed sampling: %#v", last)
			}
			log.Write([]byte("|==> | 10/128 - 512.00MB/s\r"))
			if last.Phase != "loading" {
				t.Fatalf("tensor transfer claimed encoding: %#v", last)
			}
		})
	}
}
