package llm

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/envconfig"
)

const handoffBudgetEnv = "WILDLLAMA_IMAGE_HANDOFF_MIB"
const handoffFile = "resume.bin"

type imageHandoffConfig struct {
	directory, auth string
	budget          int64
}

func newImageHandoff(launch llamaServerLaunchConfig, split bool) *imageHandoffConfig {
	if split || launch.embedding || launch.decision || launch.numParallel != 1 || launch.config.ManifestDigest == "" ||
		gpuNativeEnvironment(launch.opts, "LLAMA_ARG_SLOT_SAVE_PATH") != "" || launch.extraEnvs["LLAMA_ARG_SLOT_SAVE_PATH"] != "" {
		return nil
	}
	mib := int64(1024)
	if v, ok := os.LookupEnv(handoffBudgetEnv); ok {
		n, err := strconv.ParseInt(v, 10, 32)
		if err != nil || n <= 0 {
			return nil
		}
		mib = n
	}
	exe, err := FindLlamaServer()
	if err != nil || !NativeRuntimeSupportsEnvironment(exe, launch.gpuLibs, handoffBudgetEnv) {
		return nil
	}
	dir := filepath.Join(envconfig.Models(), ".image-handoff")
	if !filepath.IsAbs(dir) || os.MkdirAll(dir, 0700) != nil {
		return nil
	}
	cfg := &imageHandoffConfig{directory: dir, auth: rand.Text(), budget: mib << 20}
	// A service that crashed cannot restore its in-memory receipt. Only the
	// lock owner can remove a stale checkpoint, including across processes.
	if lock, err := cfg.lock(); err == nil {
		os.Remove(filepath.Join(dir, handoffFile))
		lock.Close()
	}
	return cfg
}

func (c *imageHandoffConfig) lock() (*os.File, error) {
	f, err := os.OpenFile(filepath.Join(c.directory, "handoff.lock"), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return nil, err
	}
	if err = tryHandoffLock(f); err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}

type imageReceipt struct {
	key, checksum [32]byte
	identity      nativeRuntimeFeatureIdentity
	path          string
	size          int64
	lock          *os.File
	timer         *time.Timer
}

var suspendedImagePrefix struct {
	sync.Mutex
	receipt *imageReceipt
}

func (r *imageReceipt) discard() {
	if r.timer != nil {
		r.timer.Stop()
	}
	os.Remove(r.path)
	r.lock.Close()
}

// Only immutable model manifests are eligible. Include effective native
// arguments, device IDs and runner overrides; ports and credentials vary by launch.
func (s *llamaServerRunner) imageKey() ([32]byte, nativeRuntimeFeatureIdentity, error) {
	var zero [32]byte
	id, err := nativeRuntimeIdentity(s.cmd.Path, s.launch.gpuLibs)
	if err != nil {
		return zero, id, err
	}
	args := append([]string(nil), s.cmd.Args...)
	for i := 1; i < len(args); i++ {
		if args[i-1] == "--port" {
			args[i] = ""
		}
	}
	var env []string
	for _, entry := range s.cmd.Env {
		key, _, _ := strings.Cut(entry, "=")
		if key != "WILDLLAMA_IMAGE_HANDOFF_AUTH" && (strings.HasPrefix(key, "LLAMA_ARG_") || strings.HasPrefix(key, "WILDLLAMA_") || strings.HasPrefix(key, "GGML_") || strings.HasSuffix(key, "VISIBLE_DEVICES")) {
			env = append(env, entry)
		}
	}
	sort.Strings(env)
	var devices []string
	for _, gpu := range s.launch.gpus {
		devices = append(devices, gpu.Library+"/"+gpu.ID)
	}
	data, err := json.Marshal([]any{s.launch.config, s.launch.opts.Runner, args, env, devices})
	return sha256.Sum256(data), id, err
}

func (s *llamaServerRunner) handoffAction(ctx context.Context, action string) (int64, error) {
	cfg := s.launch.handoff
	body, _ := json.Marshal(map[string]any{"filename": handoffFile, "handoff_auth": cfg.auth, "image_budget": cfg.budget})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/slots/0?action=%s", s.port, action), bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(req)
	if err != nil {
		return 0, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("slot %s returned %d", action, res.StatusCode)
	}
	var out struct {
		Written int64 `json:"n_written"`
		Read    int64 `json:"n_read"`
	}
	if err = json.NewDecoder(io.LimitReader(res.Body, 65536)).Decode(&out); err != nil {
		return 0, err
	}
	if action == "save" {
		return out.Written, nil
	}
	return out.Read, nil
}

func handoffChecksum(path string, size int64) ([32]byte, error) {
	var sum [32]byte
	f, err := os.Open(path)
	if err != nil {
		return sum, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return sum, err
	}
	if !info.Mode().IsRegular() || size <= 0 || info.Size() != size {
		return sum, errors.New("incomplete image handoff")
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, size+1))
	if err == nil && n != size {
		err = io.ErrUnexpectedEOF
	}
	copy(sum[:], h.Sum(nil))
	return sum, err
}

// Called only by the exclusive image transition after the text runner is idle.
func (s *llamaServerRunner) SaveImageHandoff(ctx context.Context) {
	cfg := s.launch.handoff
	if cfg == nil || s.cmd == nil || ctx.Err() != nil {
		return
	}
	key, id, err := s.imageKey()
	if err != nil {
		return
	}
	lock, err := cfg.lock()
	if err != nil {
		return
	}
	receipt := &imageReceipt{key: key, identity: id, path: filepath.Join(cfg.directory, handoffFile), lock: lock}
	f, err := os.OpenFile(receipt.path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0600)
	if err != nil {
		lock.Close()
		return
	}
	f.Close()
	// Let the writer finish even if the image job is canceled. Releasing the
	// file lock when HTTP is canceled could let a later job overwrite it.
	type saveResult struct {
		size     int64
		err      error
		timedOut bool
	}
	done := s.done
	result := make(chan saveResult, 1)
	go func() {
		op, cancel := context.WithTimeout(context.WithoutCancel(ctx), 90*time.Second)
		defer cancel()
		n, err := s.handoffAction(op, "save")
		result <- saveResult{n, err, op.Err() != nil}
	}()
	cleanup := func(out saveResult) {
		if out.timedOut && done != nil {
			// The native call may still have the file open.
			go func() { <-done; receipt.discard() }()
			return
		}
		receipt.discard()
	}
	var out saveResult
	select {
	case <-ctx.Done():
		go func() { cleanup(<-result) }()
		return
	case out = <-result:
	}
	n := out.size
	if out.err != nil || n <= 0 || n > cfg.budget || ctx.Err() != nil {
		cleanup(out)
		return
	}
	sum, err := handoffChecksum(receipt.path, n)
	if err != nil {
		receipt.discard()
		return
	}
	receipt.checksum, receipt.size = sum, n
	suspendedImagePrefix.Lock()
	if suspendedImagePrefix.receipt != nil {
		suspendedImagePrefix.Unlock()
		receipt.discard()
		return
	}
	suspendedImagePrefix.receipt = receipt
	receipt.timer = time.AfterFunc(30*time.Minute, func() {
		suspendedImagePrefix.Lock()
		defer suspendedImagePrefix.Unlock()
		if suspendedImagePrefix.receipt == receipt {
			suspendedImagePrefix.receipt = nil
			receipt.discard()
		}
	})
	suspendedImagePrefix.Unlock()
	slog.Info("saved text prefix for image handoff", "bytes", n)
}

func (s *llamaServerRunner) restoreImageHandoff(ctx context.Context) error {
	if s.launch.handoff == nil || s.cmd == nil {
		return nil
	}
	suspendedImagePrefix.Lock()
	receipt := suspendedImagePrefix.receipt
	if receipt == nil {
		suspendedImagePrefix.Unlock()
		return nil
	}
	key, id, err := s.imageKey()
	if err != nil || key != receipt.key || !sameNativeRuntimeIdentity(id, receipt.identity) {
		suspendedImagePrefix.Unlock()
		return nil
	}
	suspendedImagePrefix.receipt = nil
	receipt.timer.Stop()
	suspendedImagePrefix.Unlock()
	defer receipt.discard()
	sum, err := handoffChecksum(receipt.path, receipt.size)
	if err != nil || sum != receipt.checksum {
		return nil
	}
	n, err := s.handoffAction(ctx, "restore")
	if err != nil || n != receipt.size {
		// Discard partially restored native state before publishing this runner.
		_, clearErr := s.handoffAction(ctx, "erase")
		return clearErr
	}
	slog.Info("restored text prefix after image handoff", "bytes", n)
	return nil
}
