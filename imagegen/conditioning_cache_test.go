package imagegen

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func envValue(env []string, key string) string {
	for _, entry := range env {
		if value, ok := strings.CutPrefix(entry, key+"="); ok {
			return value
		}
	}
	return ""
}

func TestConditioningCacheNamespacePreservesSamplingIndependenceAndInvalidatesInputs(t *testing.T) {
	dir := t.TempDir()
	profile := testProfile(t, dir)
	config := Config{WorkDir: filepath.Join(dir, "work"), ConditioningCacheMiB: 256}
	request := Request{Prompt: "first prompt", Seed: 1, OutputPath: filepath.Join(dir, "one.png")}
	first, err := conditioningCacheEnvironment(context.Background(), config, profile, request, "native-v1")
	if err != nil {
		t.Fatal(err)
	}
	key := envValue(first, "WILDIMAGE_CONDITION_NAMESPACE")
	if len(key) != 64 || envValue(first, "WILDIMAGE_CONDITION_CACHE_BYTES") != "268435456" {
		t.Fatal("invalid native cache contract", first)
	}
	request.Seed, request.OutputPath = 999, filepath.Join(dir, "other.png")
	second, err := conditioningCacheEnvironment(context.Background(), config, profile, request, "native-v1")
	if err != nil || envValue(second, "WILDIMAGE_CONDITION_NAMESPACE") != key {
		t.Fatal("sampling-only changes invalidated deterministic encoder namespace", second, err)
	}
	for name, mutate := range map[string]func(*Profile){
		"steps":     func(p *Profile) { p.Steps++ },
		"cfg":       func(p *Profile) { p.CFGScale++ },
		"sampler":   func(p *Profile) { p.Sampler = "heun" },
		"sigmas":    func(p *Profile) { p.Sigmas = "1,0.5,0" },
		"scheduler": func(p *Profile) { p.Scheduler = "karras" },
		"sage":      func(p *Profile) { p.SageAttention = true },
	} {
		t.Run("sampling-"+name, func(t *testing.T) {
			next := profile
			mutate(&next)
			value, err := conditioningCacheEnvironment(context.Background(), config, next, request, "native-v1")
			if err != nil || envValue(value, "WILDIMAGE_CONDITION_NAMESPACE") != key {
				t.Fatal("sampling-only setting invalidated encoder namespace", value, err)
			}
		})
	}
	for name, mutate := range map[string]func(*Profile){
		"width":             func(p *Profile) { p.Width += 64 },
		"height":            func(p *Profile) { p.Height += 64 },
		"reference-size":    func(p *Profile) { p.PreserveReferenceSize = !p.PreserveReferenceSize },
		"flash-attention":   func(p *Profile) { p.FlashAttention = !p.FlashAttention },
		"threads":           func(p *Profile) { p.CPUThreads++ },
		"encoder-placement": func(p *Profile) { p.EncoderOnCPU = !p.EncoderOnCPU },
		"vae-placement":     func(p *Profile) { p.VAEOnCPU = !p.VAEOnCPU },
		"staging":           func(p *Profile) { p.StageComponents = !p.StageComponents },
		"offload":           func(p *Profile) { p.OffloadToCPU = !p.OffloadToCPU },
		"vram":              func(p *Profile) { p.MaxVRAMMiB++ },
		"ram":               func(p *Profile) { p.MaxRAMMiB++ },
		"vram-margin":       func(p *Profile) { p.VRAMReserveMiB++ },
	} {
		t.Run("conditioning-"+name, func(t *testing.T) {
			next := profile
			mutate(&next)
			value, err := conditioningCacheEnvironment(context.Background(), config, next, request, "native-v1")
			if err != nil || envValue(value, "WILDIMAGE_CONDITION_NAMESPACE") == key {
				t.Fatal("changed encoder environment reused namespace", value, err)
			}
		})
	}
	for _, mutation := range []string{"encoder", "runtime", "profile"} {
		t.Run(mutation, func(t *testing.T) {
			putModel(t, profile.TextEncoder, modelBytes())
			next := profile
			revision := "native-v1"
			switch mutation {
			case "encoder":
				putModel(t, profile.TextEncoder, append(modelBytes(), 1))
			case "runtime":
				revision = "native-v2"
			case "profile":
				next.Width = 128
			}
			value, err := conditioningCacheEnvironment(context.Background(), config, next, request, revision)
			if err != nil || envValue(value, "WILDIMAGE_CONDITION_NAMESPACE") == key {
				t.Fatal("changed encoder environment reused old namespace", value, err)
			}
		})
	}
	if _, err := os.Stat(config.WorkDir); !os.IsNotExist(err) {
		t.Fatal("preparation created cache or work directory", err)
	}
}

func TestConditioningDigestDetectsSameSizeChangesWithRestoredMtime(t *testing.T) {
	path := filepath.Join(t.TempDir(), "encoder.gguf")
	putModel(t, path, modelBytes())
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	first, err := conditioningFileIdentity(context.Background(), path, true)
	if err != nil {
		t.Fatal(err)
	}
	data := modelBytes()
	data[1024] ^= 1
	putModel(t, path, data)
	if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	second, err := conditioningFileIdentity(context.Background(), path, true)
	if err != nil || first == second {
		t.Fatal("stale encoder digest survived content change with identical size/mtime", err)
	}
}

func TestConditioningDigestMemoReusesVerifiedStampAcrossCacheInstances(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "encoder.gguf")
	putModel(t, path, modelBytes())
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	info, err := file.Stat()
	if err != nil {
		file.Close()
		t.Fatal(err)
	}
	stamp, reliable := conditioningFileStamp(file, info)
	file.Close()
	if !reliable {
		t.Skip("filesystem does not provide reliable file identities")
	}
	ctx := context.Background()
	first := openConditioningDigestMemo(dir)
	digest, err := first.fileIdentity(ctx, path, true)
	first.close(ctx)
	if err != nil {
		t.Fatal(err)
	}
	second := openConditioningDigestMemo(dir)
	defer second.close(ctx)
	if got := second.get(path, stamp); got != digest {
		t.Fatalf("persistent digest = %q, want %q", got, digest)
	}
	data := modelBytes()
	data[1024] ^= 1
	putModel(t, path, data)
	if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	changed, err := second.fileIdentity(ctx, path, true)
	if err != nil || changed == digest {
		t.Fatalf("memo reused a stale digest after a same-size change with restored mtime: %v", err)
	}
}

func TestConditioningDigestMemoBoundsStorageAndIgnoresCorruption(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	memo := openConditioningDigestMemo(dir)
	digest := conditioningDigestString([]byte("digest"))
	for index := 0; index <= conditioningDigestLimit; index++ {
		memo.put(fmt.Sprint(index), "stamp", digest)
	}
	memo.close(ctx)
	reopened := openConditioningDigestMemo(dir)
	if len(reopened.entries) != conditioningDigestLimit || reopened.get("0", "stamp") != "" || reopened.get("1", "stamp") != digest {
		t.Fatal("persistent digest memo did not retain only the bounded latest entries")
	}
	reopened.close(ctx)
	if err := os.WriteFile(filepath.Join(dir, conditioningDigestMemoName), []byte(`{"checksum":"invalid","entries":[{}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	corrupt := openConditioningDigestMemo(dir)
	defer corrupt.close(ctx)
	if len(corrupt.entries) != 0 {
		t.Fatal("persistent digest memo accepted a corrupt entry")
	}
}

func TestConditioningCacheDisabledAndInheritedEnvironmentIsolation(t *testing.T) {
	for _, limit := range []int{0, -1, 8193} {
		value, err := conditioningCacheEnvironment(context.Background(), Config{ConditioningCacheMiB: limit}, Profile{}, Request{}, "")
		if err != nil || len(value) != 0 {
			t.Fatal("disabled cache touched model files", value, err)
		}
	}
	for _, key := range []string{"WILDIMAGE_CONDITION_CACHE_DIR", "WILDIMAGE_CONDITION_NAMESPACE", "WILDIMAGE_CONDITION_CACHE_BYTES"} {
		t.Setenv(key, "untrusted-parent-value")
	}
	for _, entry := range isolatedEnvironment(Config{WorkDir: t.TempDir(), RunnerPath: filepath.Join(t.TempDir(), "sd-cli")}, t.TempDir()) {
		if strings.HasPrefix(entry, "WILDIMAGE_CONDITION_") {
			t.Fatal("inherited cache override reached image process", entry)
		}
	}
}
