package llm

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNativeRuntimeFeaturesDLLReplacement(t *testing.T) {
	dir := t.TempDir()
	executable, implementation := filepath.Join(dir, "llama-server.exe"), filepath.Join(dir, "llama-server-impl.dll")
	for _, path := range []string{executable, implementation} {
		if err := os.WriteFile(path, []byte("fixture"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	calls := 0
	probe := nativeRuntimeFeatureProbe{read: func(string, []string) ([]byte, error) {
		calls++
		if calls == 1 {
			return []byte(`{"schema":1,"environment":["OLD"]}`), nil
		}
		return []byte(`{"schema":1,"environment":["NEW"]}`), nil
	}}
	if !probe.supports(executable, nil, "OLD") {
		t.Fatal("initial runtime was not recognized")
	}
	previous, _ := os.Stat(implementation)
	replacement := implementation + ".replacement"
	if err := os.WriteFile(replacement, []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(replacement, previous.ModTime(), previous.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(implementation); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, implementation); err != nil {
		t.Fatal(err)
	}
	if !probe.supports(executable, nil, "NEW") || probe.supports(executable, nil, "OLD") || calls != 2 {
		t.Fatal("unchanged launcher retained the replaced DLL's capabilities")
	}
}

func TestNativeRuntimeFeaturesLibraryContexts(t *testing.T) {
	executable := featureTestFile(t)
	calls := 0
	probe := nativeRuntimeFeatureProbe{read: func(_ string, libraries []string) ([]byte, error) {
		calls++
		if len(libraries) > 0 && libraries[0] == "first" {
			return []byte(`{"schema":1,"environment":["FIRST"]}`), nil
		}
		return []byte(`{"schema":1,"environment":["SECOND"]}`), nil
	}}
	first, second := []string{"first", "second"}, []string{"second", "first"}
	for range 3 {
		if !probe.supports(executable, first, "FIRST") || probe.supports(executable, first, "SECOND") || !probe.supports(executable, second, "SECOND") || probe.supports(executable, second, "FIRST") {
			t.Fatal("different library search orders shared a capability cache entry")
		}
	}
	if calls != 2 {
		t.Fatalf("stable library contexts queried %d times", calls)
	}
}

func TestNativeRuntimeFeaturesChangingDLLRejected(t *testing.T) {
	dir := t.TempDir()
	executable, implementation := filepath.Join(dir, "llama-server.exe"), filepath.Join(dir, "llama-server-impl.dll")
	for _, path := range []string{executable, implementation} {
		if err := os.WriteFile(path, []byte("fixture"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	probe := nativeRuntimeFeatureProbe{read: func(string, []string) ([]byte, error) {
		if err := os.WriteFile(implementation, []byte("a different runtime"), 0600); err != nil {
			return nil, err
		}
		return []byte(`{"schema":1,"environment":["VISION"]}`), nil
	}}
	if probe.supports(executable, nil, "VISION") {
		t.Fatal("a DLL changed during the feature query but its response was accepted")
	}
}

func TestNativeRuntimeFeaturesPortableCompanionIdentity(t *testing.T) {
	for _, platform := range []struct{ goos, launcher, library string }{
		{"windows", "llama-server.exe", "llama-server-impl.dll"},
		{"linux", "llama-server", "libllama-server-impl.so"},
		{"darwin", "llama-server", "libllama-server-impl.dylib"},
	} {
		t.Run(platform.goos, func(t *testing.T) {
			dir, fallback := t.TempDir(), t.TempDir()
			executable := filepath.Join(dir, platform.launcher)
			if name := nativeRuntimeImplementationName(executable, platform.goos); name != platform.library {
				t.Fatalf("implementation=%q want=%q", name, platform.library)
			}
			if err := os.WriteFile(executable, []byte("launcher"), 0600); err != nil {
				t.Fatal(err)
			}
			library := filepath.Join(fallback, platform.library)
			if err := os.WriteFile(library, []byte("first"), 0600); err != nil {
				t.Fatal(err)
			}
			paths := []string{dir, fallback}
			before, err := nativeRuntimeIdentityInPaths(executable, paths, platform.library)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, platform.library), []byte("newer"), 0600); err != nil {
				t.Fatal(err)
			}
			after, err := nativeRuntimeIdentityInPaths(executable, paths, platform.library)
			if err != nil || sameNativeRuntimeIdentity(before, after) {
				t.Fatal("higher-priority implementation did not invalidate capabilities")
			}
			if err := os.Remove(filepath.Join(dir, platform.library)); err != nil {
				t.Fatal(err)
			}
			after, err = nativeRuntimeIdentityInPaths(executable, paths, platform.library)
			if err != nil || !sameNativeRuntimeIdentity(before, after) {
				t.Fatal("unchanged fallback implementation did not recover its identity")
			}
			if err := os.Remove(library); err != nil {
				t.Fatal(err)
			}
			after, err = nativeRuntimeIdentityInPaths(executable, paths, platform.library)
			if err != nil || sameNativeRuntimeIdentity(before, after) {
				t.Fatal("removed implementation did not invalidate capabilities")
			}
		})
	}
}
