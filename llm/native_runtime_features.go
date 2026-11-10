package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"

	"github.com/ollama/ollama/ml"
)

const nativeRuntimeFeatureOutputLimit = 32 << 10

type nativeRuntimeFeatureEntry struct {
	identity    nativeRuntimeFeatureIdentity
	environment map[string]bool
}

type nativeRuntimeFeatureIdentity struct {
	executable     os.FileInfo
	librarySearch  string
	implementation []nativeRuntimeFeatureFile
}

type nativeRuntimeFeatureFile struct {
	path string
	info os.FileInfo
}

type nativeRuntimeFeatureProbe struct {
	mu      sync.Mutex
	entries map[string]nativeRuntimeFeatureEntry
	flights singleflight.Group
	read    func(string, []string) ([]byte, error)
}

var nativeRuntimeFeatures = nativeRuntimeFeatureProbe{read: readNativeRuntimeFeatures}

// NativeRuntimeSupportsEnvironment queries the selected executable, without
// opening a model. Old or unavailable payloads retain conventional behavior.
func NativeRuntimeSupportsEnvironment(executable string, libraries []string, key string) bool {
	return nativeRuntimeFeatures.supports(executable, libraries, key)
}

func sameNativeRuntimeFeatureFile(a, b os.FileInfo) bool {
	return a != nil && b != nil && a.Mode().IsRegular() && b.Mode().IsRegular() &&
		os.SameFile(a, b) && a.Size() == b.Size() && a.ModTime().Equal(b.ModTime())
}

func nativeRuntimeImplementationName(executable, goos string) string {
	if strings.EqualFold(filepath.Base(executable), "llama-server.exe") {
		return "llama-server-impl.dll"
	}
	if filepath.Base(executable) == "llama-server" {
		switch goos {
		case "windows":
			return "llama-server-impl.dll"
		case "darwin":
			return "libllama-server-impl.dylib"
		}
		return "libllama-server-impl.so"
	}
	return ""
}

func nativeRuntimeIdentity(executable string, libraries []string) (nativeRuntimeFeatureIdentity, error) {
	name := nativeRuntimeImplementationName(executable, runtime.GOOS)
	if name == "" {
		// Custom/self-contained executables have no known companion name.
		return nativeRuntimeIdentityInPaths(executable, libraries, "")
	}
	var paths []string
	seen := make(map[string]bool)
	add := func(path string) {
		if path == "" {
			return
		}
		if absolute, err := filepath.Abs(path); err == nil {
			path = absolute
		}
		if !seen[path] {
			seen[path] = true
			paths = append(paths, path)
		}
	}
	add(filepath.Dir(executable))
	for _, path := range libraries {
		if path != ml.LibOllamaPath {
			add(path)
		}
	}
	for _, path := range filepath.SplitList(os.Getenv(llamaServerLibraryPathEnv())) {
		add(path)
	}
	return nativeRuntimeIdentityInPaths(executable, paths, name)
}

func nativeRuntimeIdentityInPaths(executable string, paths []string, implementationName string) (nativeRuntimeFeatureIdentity, error) {
	info, err := os.Stat(executable)
	if err != nil {
		return nativeRuntimeFeatureIdentity{}, err
	}
	if !info.Mode().IsRegular() {
		return nativeRuntimeFeatureIdentity{}, fmt.Errorf("native runtime is not a regular file")
	}
	identity := nativeRuntimeFeatureIdentity{executable: info, librarySearch: strings.Join(paths, "\x00")}
	if implementationName == "" {
		return identity, nil
	}
	// Shared Windows, Linux and macOS builds use a small launcher. Track the
	// selected implementation and earlier missing candidates so appearance or
	// replacement of a higher-priority companion invalidates the cache too.
	for _, path := range paths {
		candidate := filepath.Join(path, implementationName)
		implementation, err := os.Stat(candidate)
		if err != nil && !os.IsNotExist(err) {
			return nativeRuntimeFeatureIdentity{}, err
		}
		if os.IsNotExist(err) {
			identity.implementation = append(identity.implementation, nativeRuntimeFeatureFile{path: candidate})
			continue
		}
		if err == nil {
			if !implementation.Mode().IsRegular() {
				return nativeRuntimeFeatureIdentity{}, fmt.Errorf("native runtime implementation is not a regular file")
			}
			identity.implementation = append(identity.implementation, nativeRuntimeFeatureFile{path: candidate, info: implementation})
			break
		}
	}
	return identity, nil
}

func sameNativeRuntimeIdentity(a, b nativeRuntimeFeatureIdentity) bool {
	if !sameNativeRuntimeFeatureFile(a.executable, b.executable) || a.librarySearch != b.librarySearch || len(a.implementation) != len(b.implementation) {
		return false
	}
	for i, file := range a.implementation {
		other := b.implementation[i]
		if file.path != other.path || !(file.info == nil && other.info == nil || sameNativeRuntimeFeatureFile(file.info, other.info)) {
			return false
		}
	}
	return true
}

func (p *nativeRuntimeFeatureProbe) supports(executable string, libraries []string, key string) bool {
	executable, err := filepath.Abs(executable)
	if err != nil {
		return false
	}
	identity, err := nativeRuntimeIdentity(executable, libraries)
	if err != nil {
		return false
	}
	// Search order is part of the native process environment; two backends using
	// the same launcher must not borrow one another's capability response.
	cacheKey := executable + "\x00" + identity.librarySearch
	cached := func() (nativeRuntimeFeatureEntry, bool) {
		p.mu.Lock()
		defer p.mu.Unlock()
		entry, ok := p.entries[cacheKey]
		return entry, ok && sameNativeRuntimeIdentity(entry.identity, identity)
	}
	if entry, ok := cached(); ok {
		return entry.environment[key]
	}
	value, err, _ := p.flights.Do(cacheKey, func() (any, error) {
		if entry, ok := cached(); ok {
			return entry, nil
		}
		output, err := p.read(executable, libraries)
		if err != nil || len(output) > nativeRuntimeFeatureOutputLimit {
			return nil, fmt.Errorf("native runtime feature query unavailable")
		}
		var response struct {
			Schema      int      `json:"schema"`
			Environment []string `json:"environment"`
		}
		if err := json.Unmarshal(output, &response); err != nil || response.Schema != 1 || response.Environment == nil {
			return nil, fmt.Errorf("unsupported native runtime feature response")
		}
		environment := make(map[string]bool, len(response.Environment))
		for _, name := range response.Environment {
			environment[name] = true
		}
		current, err := nativeRuntimeIdentity(executable, libraries)
		if err != nil || !sameNativeRuntimeIdentity(identity, current) {
			return nil, fmt.Errorf("native runtime changed during feature query")
		}
		p.mu.Lock()
		if p.entries == nil {
			p.entries = make(map[string]nativeRuntimeFeatureEntry)
		}
		// The process normally uses one payload; keep fallback/test paths bounded.
		if len(p.entries) >= 16 {
			clear(p.entries)
		}
		entry := nativeRuntimeFeatureEntry{identity: current, environment: environment}
		p.entries[cacheKey] = entry
		p.mu.Unlock()
		return entry, nil
	})
	if err != nil {
		return false
	}
	entry := value.(nativeRuntimeFeatureEntry)
	// A concurrent waiter can join a probe that started before a DLL swap.
	// Never enable a feature from that superseded payload.
	current, err := nativeRuntimeIdentity(executable, libraries)
	return err == nil && sameNativeRuntimeIdentity(entry.identity, current) && entry.environment[key]
}

// Drain a faulty/old payload's output without retaining an unbounded help dump.
type nativeFeatureOutput struct {
	buffer    bytes.Buffer
	truncated bool
}

func (w *nativeFeatureOutput) Write(data []byte) (int, error) {
	size := len(data)
	remaining := nativeRuntimeFeatureOutputLimit - w.buffer.Len()
	if size > remaining {
		w.truncated = true
		data = data[:remaining]
	}
	_, _ = w.buffer.Write(data)
	return size, nil
}

func readNativeRuntimeFeatures(executable string, libraries []string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, executable, "--wildllama-runtime-features")
	cmd.SysProcAttr = LlamaServerSysProcAttr
	cmd.WaitDelay = 250 * time.Millisecond
	SetupLlamaServerCommandEnv(cmd, executable, libraries, nil)
	var output nativeFeatureOutput
	cmd.Stdout = &output
	if err := cmd.Run(); err != nil {
		return nil, err
	}
	if output.truncated {
		return nil, fmt.Errorf("native runtime feature output exceeds limit")
	}
	return output.buffer.Bytes(), nil
}
