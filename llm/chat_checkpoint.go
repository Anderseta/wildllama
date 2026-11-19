package llm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/version"
)

const checkpointBudgetEnv = "WILDLLAMA_CHAT_CHECKPOINT_MIB"

type chatCheckpoint struct {
	directory, owner, runtime string
	budget, maxFile           int64
}

func ValidCheckpointSession(id string) bool {
	if len(id) == 0 || len(id) > 128 {
		return false
	}
	for _, b := range []byte(id) {
		if !(b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' || b == '_' || b == '-' || b == '.') {
			return false
		}
	}
	return true
}

func checkpointBudget() int64 {
	if raw, ok := os.LookupEnv(checkpointBudgetEnv); ok {
		n, err := strconv.ParseInt(raw, 10, 32)
		if err != nil || n <= 0 {
			return 0
		}
		return n << 20
	}
	return 2048 << 20
}

func (s *llamaServerRunner) checkpoint(id string) *chatCheckpoint {
	cfg := s.launch.handoff
	if cfg == nil || s.cmd == nil || !ValidCheckpointSession(id) {
		return nil
	}
	limit := checkpointBudget()
	if limit <= 0 {
		return nil
	}
	if s.checkpointRuntime == "" {
		runtime, err := s.checkpointRuntimeKey()
		if err != nil {
			return nil
		}
		s.checkpointRuntime = runtime
	}
	return &chatCheckpoint{directory: filepath.Join(filepath.Dir(cfg.directory), ".chat-checkpoints"),
		owner: fmt.Sprintf("%x", sha256.Sum256([]byte(id))), runtime: s.checkpointRuntime,
		budget: limit, maxFile: min(cfg.budget, limit/3)}
}

// A runner's loaded weights, libraries and launch configuration are immutable.
// Fingerprint them once per runner so ordinary turns never scan installation directories.
func (s *llamaServerRunner) checkpointRuntimeKey() (string, error) {
	key, identity, err := s.imageKey()
	if err != nil {
		return "", err
	}
	hash := sha256.New()
	hash.Write(key[:])
	fmt.Fprintf(hash, "\x00%s\x00%s:%d:%d", version.Version, s.cmd.Path, identity.executable.Size(), identity.executable.ModTime().UnixNano())
	seen := map[string]bool{}
	dirs := append([]string{filepath.Dir(s.cmd.Path)}, s.launch.gpuLibs...)
	for _, lib := range identity.implementation {
		if lib.info != nil {
			dirs = append(dirs, filepath.Dir(lib.path))
		}
	}
	for _, dir := range dirs {
		if dir == "" || seen[dir] {
			continue
		}
		seen[dir] = true
		entries, err := os.ReadDir(dir)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return "", err
		}
		for _, entry := range entries {
			name := strings.ToLower(entry.Name())
			base := strings.TrimPrefix(name, "lib")
			if !(strings.HasPrefix(base, "llama") || strings.HasPrefix(base, "ggml") || strings.HasPrefix(base, "mtmd")) ||
				!(strings.HasSuffix(name, ".dll") || strings.HasSuffix(name, ".dylib") || strings.HasSuffix(name, ".so") || strings.Contains(name, ".so.")) {
				continue
			}
			path := filepath.Join(dir, entry.Name())
			info, err := os.Stat(path)
			if err != nil {
				return "", err
			}
			if info.Mode().IsRegular() {
				fmt.Fprintf(hash, "\x00%s:%d:%d", path, info.Size(), info.ModTime().UnixNano())
			}
		}
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func (c *chatCheckpoint) prefix() string { return c.owner + "." + c.runtime + "." }
func (c *chatCheckpoint) revoked() bool {
	_, err := os.Lstat(filepath.Join(c.directory, c.owner+".deleted"))
	return !errors.Is(err, os.ErrNotExist)
}

type checkpointFile struct {
	name     string
	size     int64
	modified time.Time
}

func checkpointName(name string) bool {
	if len(name) != 64*3+3+4 || !strings.HasSuffix(name, ".slot") || name[64] != '.' || name[129] != '.' {
		return false
	}
	for _, part := range []string{name[:64], name[65:129], name[130:194]} {
		if _, err := hex.DecodeString(part); err != nil {
			return false
		}
	}
	return true
}
func (c *chatCheckpoint) files() ([]checkpointFile, int64, error) {
	info, err := os.Lstat(c.directory)
	if err != nil {
		return nil, 0, err
	}
	if !info.IsDir() {
		return nil, 0, fmt.Errorf("invalid checkpoint directory")
	}
	list, err := os.ReadDir(c.directory)
	if err != nil {
		return nil, 0, err
	}
	var files []checkpointFile
	var total int64
	for _, entry := range list {
		if !checkpointName(entry.Name()) || !entry.Type().IsRegular() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return nil, 0, err
		}
		total += info.Size()
		files = append(files, checkpointFile{entry.Name(), info.Size(), info.ModTime()})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].modified.After(files[j].modified) })
	return files, total, nil
}

// The runner semaphore protects the single native slot. The interprocess handoff
// lock protects its fixed scratch filename across daemon and image transitions.
// A timeout may leave the native writer running until the process exits.
func (s *llamaServerRunner) checkpointIO(ctx context.Context, action string, maxFile int64, lock *os.File, scratch string) (int64, error) {
	child, cancel := context.WithTimeout(context.WithoutCancel(ctx), 90*time.Second)
	defer cancel()
	n, err := s.handoffActionBudget(child, action, maxFile)
	if err != nil && child.Err() != nil && s.done != nil {
		go func() { <-s.done; os.Remove(scratch); lock.Close() }()
		return 0, errCheckpointWriterActive
	}
	return n, err
}

var errCheckpointWriterActive = errors.New("checkpoint operation is still finishing")

func (s *llamaServerRunner) beginChatCheckpoint(ctx context.Context, id string) (*chatCheckpoint, error) {
	if s.launch.handoff == nil {
		return nil, nil
	}
	c := s.checkpoint(id)
	previous := s.checkpointResident
	s.checkpointResident = ""
	if c == nil || c.revoked() {
		return nil, nil
	}
	// A missing cache root is normal on the first conversation.
	if previous == c.prefix() {
		s.checkpointResident = previous
		return c, nil
	}
	s.checkpointTime = time.Time{}
	s.checkpointTokens = 0
	cfg := s.launch.handoff
	lock, err := cfg.lock()
	if err != nil {
		return c, nil
	}
	scratch := filepath.Join(cfg.directory, handoffFile)
	release := true
	defer func() {
		if release {
			os.Remove(scratch)
			lock.Close()
		}
	}()
	files, _, err := c.files()
	if err != nil {
		return c, nil
	}
	for _, f := range files {
		if !strings.HasPrefix(f.name, c.prefix()) || f.size <= 0 || f.size > c.maxFile {
			continue
		}
		source := filepath.Join(c.directory, f.name)
		sum, err := handoffChecksum(source, f.size)
		if err != nil || hex.EncodeToString(sum[:]) != strings.TrimSuffix(strings.TrimPrefix(f.name, c.prefix()), ".slot") {
			os.Remove(source)
			continue
		}
		if c.revoked() {
			return nil, nil
		}
		os.Remove(scratch)
		if os.Link(source, scratch) != nil {
			return c, nil
		}
		n, err := s.checkpointIO(ctx, "restore", c.maxFile, lock, scratch)
		if errors.Is(err, errCheckpointWriterActive) {
			release = false
			return nil, err
		}
		if err != nil || n != f.size {
			os.Remove(source)
			if _, clearErr := s.handoffAction(ctx, "erase"); clearErr != nil {
				return nil, fmt.Errorf("reset failed chat restore: %w", clearErr)
			}
			return c, nil
		}
		now := time.Now()
		os.Chtimes(source, now, now)
		slog.Debug("restored chat prefix", "bytes", n)
		break
	}
	return c, nil
}

func (s *llamaServerRunner) finishChatCheckpoint(ctx context.Context, c *chatCheckpoint, tokens int) {
	s.checkpointResident = c.prefix()
	if tokens < 1024 || c.maxFile <= 0 || c.revoked() {
		return
	}
	if !s.checkpointTime.IsZero() && time.Since(s.checkpointTime) < 5*time.Minute && tokens < s.checkpointTokens+2048 {
		return
	}
	cfg := s.launch.handoff
	lock, err := cfg.lock()
	if err != nil {
		return
	}
	scratch := filepath.Join(cfg.directory, handoffFile)
	release := true
	defer func() {
		if release {
			os.Remove(scratch)
			lock.Close()
		}
	}()
	if os.MkdirAll(c.directory, 0700) != nil || c.revoked() {
		return
	}
	files, used, err := c.files()
	if err != nil {
		return
	}
	// Keep space for a new snapshot without exceeding the total disk budget.
	for i := len(files) - 1; used > c.budget-c.maxFile && i >= 0; i-- {
		if os.Remove(filepath.Join(c.directory, files[i].name)) != nil {
			return
		}
		used -= files[i].size
	}
	os.Remove(scratch)
	f, err := os.OpenFile(scratch, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return
	}
	f.Close()
	s.checkpointTime = time.Now()
	s.checkpointTokens = tokens
	n, err := s.checkpointIO(ctx, "save", c.maxFile, lock, scratch)
	if errors.Is(err, errCheckpointWriterActive) {
		release = false
		return
	}
	if err != nil || n <= 0 || n > c.maxFile || c.revoked() {
		return
	}
	sum, err := handoffChecksum(scratch, n)
	if err != nil {
		return
	}
	name := c.prefix() + hex.EncodeToString(sum[:]) + ".slot"
	dest := filepath.Join(c.directory, name)
	if err = os.Rename(scratch, dest); err != nil {
		return
	}
	// Offline deletion publishes the revocation before it enumerates slots.
	if c.revoked() {
		os.Remove(dest)
		return
	}
	for _, f := range files {
		if f.name != name && strings.HasPrefix(f.name, c.prefix()) {
			os.Remove(filepath.Join(c.directory, f.name))
		}
	}
	slog.Debug("saved chat prefix", "bytes", n)
}
