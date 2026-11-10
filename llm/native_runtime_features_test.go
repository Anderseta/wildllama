package llm

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func init() {
	if len(os.Args) != 2 || os.Args[1] != "--wildllama-runtime-features" {
		return
	}
	switch os.Getenv("OLLAMA_TEST_RUNTIME_FEATURE_HELPER_MODE") {
	case "valid":
		fmt.Println(`{"schema":1,"environment":["WILDLLAMA_MMPROJ_ON_DEMAND"]}`)
		os.Exit(0)
	case "oversized":
		fmt.Print(strings.Repeat("x", nativeRuntimeFeatureOutputLimit*2))
		os.Exit(0)
	case "timeout":
		time.Sleep(10 * time.Second)
		os.Exit(0)
	}
}

func featureTestFile(t *testing.T) string {
	t.Helper()
	file := filepath.Join(t.TempDir(), "native-payload")
	if err := os.WriteFile(file, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	return file
}

func TestNativeRuntimeFeaturesCacheAndFileReplacement(t *testing.T) {
	file := featureTestFile(t)
	var calls atomic.Int32
	probe := nativeRuntimeFeatureProbe{read: func(string, []string) ([]byte, error) {
		calls.Add(1)
		return []byte(`{"schema":1,"environment":["VISION"]}`), nil
	}}
	for range 3 {
		if !probe.supports(file, nil, "VISION") || probe.supports(file, nil, "UNKNOWN") {
			t.Fatal("feature discovery or cached unsupported feature differs")
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("unchanged payload queried %d times", calls.Load())
	}
	previous, _ := os.Stat(file)
	replacement := file + ".replacement"
	if err := os.WriteFile(replacement, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(replacement, previous.ModTime(), previous.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, file); err != nil {
		t.Fatal(err)
	}
	if !probe.supports(file, nil, "VISION") || calls.Load() != 2 {
		t.Fatal("replacement with same size/time retained old capabilities")
	}
}

func TestNativeRuntimeFeaturesRejectUnknownMalformedAndChangingPayloads(t *testing.T) {
	for name, output := range map[string]string{
		"unknown schema": `{"schema":2,"environment":["VISION"]}`,
		"missing list":   `{"schema":1}`,
		"invalid json":   `partial`,
		"oversized":      strings.Repeat("x", nativeRuntimeFeatureOutputLimit+1),
	} {
		t.Run(name, func(t *testing.T) {
			probe := nativeRuntimeFeatureProbe{read: func(string, []string) ([]byte, error) { return []byte(output), nil }}
			if probe.supports(featureTestFile(t), nil, "VISION") {
				t.Fatal("invalid feature response enabled projector lifecycle")
			}
		})
	}
	file := featureTestFile(t)
	probe := nativeRuntimeFeatureProbe{read: func(path string, _ []string) ([]byte, error) {
		if err := os.WriteFile(path, []byte("changed-payload"), 0600); err != nil {
			return nil, err
		}
		return []byte(`{"schema":1,"environment":["VISION"]}`), nil
	}}
	if probe.supports(file, nil, "VISION") {
		t.Fatal("capabilities from a changing payload were accepted")
	}
}

func TestNativeRuntimeFeaturesCoalescesConcurrentQueries(t *testing.T) {
	file := featureTestFile(t)
	var calls atomic.Int32
	probe := nativeRuntimeFeatureProbe{read: func(string, []string) ([]byte, error) {
		calls.Add(1)
		time.Sleep(20 * time.Millisecond)
		return []byte(`{"schema":1,"environment":["VISION"]}`), nil
	}}
	var tasks sync.WaitGroup
	for range 12 {
		tasks.Go(func() {
			if !probe.supports(file, nil, "VISION") {
				t.Error("concurrent feature discovery failed")
			}
		})
	}
	tasks.Wait()
	if calls.Load() != 1 {
		t.Fatalf("concurrent loads queried native payload %d times", calls.Load())
	}
}

func TestNativeRuntimeFeaturesFailureCanRecover(t *testing.T) {
	file := featureTestFile(t)
	var calls int
	probe := nativeRuntimeFeatureProbe{read: func(string, []string) ([]byte, error) {
		calls++
		if calls == 1 {
			return nil, errors.New("temporary native payload failure")
		}
		return []byte(`{"schema":1,"environment":[]}`), nil
	}}
	if probe.supports(file, nil, "VISION") || probe.supports(file, nil, "VISION") || calls != 2 {
		t.Fatal("failed query did not recover to a cached unsupported feature")
	}
	_ = probe.supports(file, nil, "VISION")
	if calls != 2 {
		t.Fatal("valid old-payload response was not cached")
	}
}

func TestNativeFeatureOutputBound(t *testing.T) {
	var output nativeFeatureOutput
	data := bytes.Repeat([]byte("x"), nativeRuntimeFeatureOutputLimit*3)
	if n, err := output.Write(data); err != nil || n != len(data) || !output.truncated || output.buffer.Len() != nativeRuntimeFeatureOutputLimit {
		t.Fatal("native feature output was not bounded and drained")
	}
	if n, err := output.Write([]byte("more")); err != nil || n != 4 || output.buffer.Len() != nativeRuntimeFeatureOutputLimit {
		t.Fatal("full native feature buffer stopped draining output")
	}
}

func TestNativeRuntimeFeaturesSubprocessBoundaries(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_TEST_RUNTIME_FEATURE_HELPER_MODE", "valid")
	output, err := readNativeRuntimeFeatures(executable, nil)
	if err != nil || !bytes.Contains(output, []byte(projectorOnDemandEnvironment)) {
		t.Fatalf("native feature subprocess failed: %q %v", output, err)
	}
	t.Setenv("OLLAMA_TEST_RUNTIME_FEATURE_HELPER_MODE", "oversized")
	if _, err := readNativeRuntimeFeatures(executable, nil); err == nil {
		t.Fatal("oversized subprocess output accepted")
	}
	t.Setenv("OLLAMA_TEST_RUNTIME_FEATURE_HELPER_MODE", "timeout")
	start := time.Now()
	if _, err := readNativeRuntimeFeatures(executable, nil); err == nil {
		t.Fatal("native feature subprocess did not time out")
	}
	if time.Since(start) > 5*time.Second {
		t.Fatal("feature query did not terminate within its bounded startup delay")
	}
}
