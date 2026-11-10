package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
)

const nativeRuntimeFeatureOutputLimit = 32 << 10

type nativeRuntimeFeatureEntry struct {
	info        os.FileInfo
	environment map[string]bool
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

func (p *nativeRuntimeFeatureProbe) supports(executable string, libraries []string, key string) bool {
	executable, err := filepath.Abs(executable)
	if err != nil {
		return false
	}
	info, err := os.Stat(executable)
	if err != nil || !info.Mode().IsRegular() {
		return false
	}
	cached := func() (map[string]bool, bool) {
		p.mu.Lock()
		defer p.mu.Unlock()
		entry, ok := p.entries[executable]
		return entry.environment, ok && sameNativeRuntimeFeatureFile(entry.info, info)
	}
	if environment, ok := cached(); ok {
		return environment[key]
	}
	identity := fmt.Sprintf("%s:%d:%d", executable, info.Size(), info.ModTime().UnixNano())
	value, err, _ := p.flights.Do(identity, func() (any, error) {
		if environment, ok := cached(); ok {
			return environment, nil
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
		current, err := os.Stat(executable)
		if err != nil || !sameNativeRuntimeFeatureFile(info, current) {
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
		p.entries[executable] = nativeRuntimeFeatureEntry{info: current, environment: environment}
		p.mu.Unlock()
		return environment, nil
	})
	if err != nil {
		return false
	}
	return value.(map[string]bool)[key]
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
