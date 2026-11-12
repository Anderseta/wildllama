package imagegen

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type conditioningDigestEntry struct {
	stamp string
	hash  string
	used  uint64
}

var conditioningDigests = struct {
	sync.Mutex
	entries map[string]conditioningDigestEntry
	clock   uint64
}{entries: make(map[string]conditioningDigestEntry)}

// Cache fingerprints include the filesystem's change time and file identity,
// so modifying weights invalidates them even when mtime is restored manually.
func conditioningFileIdentity(ctx context.Context, path string, content bool, reports ...func(int64, int64)) (string, error) {
	return (*conditioningDigestMemo)(nil).fileIdentity(ctx, path, content, reports...)
}

func (memo *conditioningDigestMemo) fileIdentity(ctx context.Context, path string, content bool, reports ...func(int64, int64)) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return "", errors.New("conditioning component must be a regular file")
	}
	stamp, reliable := conditioningFileStamp(f, info)
	if reliable && !content {
		return stamp, nil
	}
	if reliable {
		conditioningDigests.Lock()
		entry, ok := conditioningDigests.entries[path]
		if ok && entry.stamp == stamp {
			conditioningDigests.clock++
			entry.used = conditioningDigests.clock
			conditioningDigests.entries[path] = entry
			conditioningDigests.Unlock()
			return entry.hash, nil
		}
		conditioningDigests.Unlock()
		if digest := memo.get(path, stamp); digest != "" {
			return digest, nil
		}
	}
	hash := sha256.New()
	buffer := make([]byte, 1<<20)
	var copied int64
	lastReport := time.Now()
	if len(reports) > 0 && reports[0] != nil {
		reports[0](0, info.Size())
	}
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, readErr := f.Read(buffer)
		if n > 0 {
			_, _ = hash.Write(buffer[:n])
			copied += int64(n)
			if len(reports) > 0 && reports[0] != nil && (copied == info.Size() || time.Since(lastReport) >= 250*time.Millisecond) {
				reports[0](copied, info.Size())
				lastReport = time.Now()
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return "", readErr
		}
	}
	after, err := f.Stat()
	if err != nil {
		return "", err
	}
	afterStamp, afterReliable := conditioningFileStamp(f, after)
	if info.Size() != after.Size() || !info.ModTime().Equal(after.ModTime()) || reliable && (!afterReliable || afterStamp != stamp) {
		return "", errors.New("conditioning component changed while hashing")
	}
	digest := hex.EncodeToString(hash.Sum(nil))
	if reliable {
		conditioningDigests.Lock()
		conditioningDigests.clock++
		conditioningDigests.entries[path] = conditioningDigestEntry{stamp: stamp, hash: digest, used: conditioningDigests.clock}
		if len(conditioningDigests.entries) > 64 {
			oldest, used := "", ^uint64(0)
			for name, entry := range conditioningDigests.entries {
				if entry.used < used {
					oldest, used = name, entry.used
				}
			}
			delete(conditioningDigests.entries, oldest)
		}
		conditioningDigests.Unlock()
		memo.put(path, stamp, digest)
	}
	return digest, nil
}

func conditioningCacheEnvironment(ctx context.Context, config Config, profile Profile, request Request, runtimeRevision string, reports ...func(Progress)) ([]string, error) {
	if config.ConditioningCacheMiB <= 0 || config.ConditioningCacheMiB > 8192 {
		return nil, nil
	}
	directory := filepath.Join(config.WorkDir, "conditioning-cache")
	if err := validateWritePath(directory); err != nil {
		return nil, err
	}
	// Never write through a cache symlink that escapes the configured work root.
	root, err := resolvedWritePath(config.WorkDir)
	if err != nil {
		return nil, err
	}
	resolved, err := resolvedWritePath(directory)
	if err != nil || !within(root, resolved) {
		return nil, errors.New("conditioning cache resolves outside the work directory")
	}
	memo := openConditioningDigestMemo(root)
	defer memo.close(ctx)
	components := make(map[string]string)
	if config.RunnerPath != "" {
		// The pinned manifest contains the native DLL, patch and header hashes.
		// A rebuilt runtime therefore cannot reuse old embeddings accidentally,
		// even when the upstream revision string itself has not changed.
		identity, err := conditioningFileIdentity(ctx, filepath.Join(filepath.Dir(config.RunnerPath), "wildimage-runtime.json"), true)
		if err != nil {
			return nil, err
		}
		components["runtime-manifest"] = identity
	}
	report := func(current, total int64) {
		if len(reports) == 0 || reports[0] == nil {
			return
		}
		// MiB counters also remain representable on 32-bit hosts.
		end := int((total + (1 << 20) - 1) >> 20)
		now := int(current >> 20)
		if current == total {
			now = end
		}
		reports[0](Progress{Phase: "conditioning_cache_prepare", Message: "Checking prompt cache model signatures", Current: now, Total: end})
	}
	for role, path := range map[string]string{"diffusion": profile.DiffusionModel, "encoder": profile.TextEncoder, "vae": profile.VAE} {
		identity, err := memo.fileIdentity(ctx, path, role == "encoder", report)
		if err != nil {
			return nil, err
		}
		components[role] = identity
	}
	if len(request.ReferenceImages) > 0 && !sameComponentPath(profile.VisionEncoder, profile.TextEncoder) {
		identity, err := memo.fileIdentity(ctx, profile.VisionEncoder, true, report)
		if err != nil {
			return nil, err
		}
		components["vision"] = identity
	}
	for i, lora := range profile.LoRAs {
		if lora.Scale == 0 {
			continue
		}
		identity, err := memo.fileIdentity(ctx, lora.Path, true, report)
		if err != nil {
			return nil, err
		}
		components[fmt.Sprintf("lora-%d", i)] = identity
	}
	for i, path := range request.ReferenceImages {
		identity, err := conditioningFileIdentity(ctx, path, true, report)
		if err != nil {
			return nil, err
		}
		components[fmt.Sprintf("reference-%d", i)] = identity
	}
	// Sampling seed/output path do not affect deterministic prompt encoding.
	// Native code additionally hashes every actual ConditionerParams field and
	// reference tensor before accepting any serialized conditioning result.
	conditioningProfile := profile
	conditioningProfile.Steps = 0
	conditioningProfile.CFGScale = 0
	conditioningProfile.Sampler = ""
	conditioningProfile.Sigmas = ""
	conditioningProfile.Scheduler = ""
	conditioningProfile.SageAttention = false // Applies only to the diffusion model.
	// Admission clamps these to live free memory on every request. They bound
	// allocations, not the encoder math, and must not invalidate identical input
	// when unrelated desktop/GPU activity changes. The native cache loader still
	// enforces the current RAM budget; CPU retries use a different placement key.
	conditioningProfile.MaxVRAMMiB = 0
	conditioningProfile.MaxRAMMiB = 0
	conditioningProfile.VRAMReserveMiB = 0
	conditioningProfile.ID = ""
	conditioningProfile.ModelID = ""
	encoded, err := json.Marshal(struct {
		Schema     int               `json:"schema"`
		Runtime    string            `json:"runtime"`
		Profile    Profile           `json:"profile"`
		Components map[string]string `json:"components"`
	}{1, runtimeRevision, conditioningProfile, components})
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(encoded)
	return []string{
		"WILDIMAGE_CONDITION_CACHE_DIR=" + directory,
		"WILDIMAGE_CONDITION_NAMESPACE=" + hex.EncodeToString(digest[:]),
		fmt.Sprintf("WILDIMAGE_CONDITION_CACHE_BYTES=%d", int64(config.ConditioningCacheMiB)<<20),
	}, nil
}
