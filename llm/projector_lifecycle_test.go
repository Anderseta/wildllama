package llm

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestProjectorLifecycleEnvironmentRequiresAdvertisedProjectorSupport(t *testing.T) {
	t.Setenv(projectorOnDemandEnvironment, "1")
	for _, test := range []struct {
		name       string
		projectors []string
		supported  bool
		want       string
	}{
		{"no projector", nil, true, ""},
		{"old payload", []string{"vision.gguf"}, false, ""},
		{"supported payload", []string{"vision.gguf"}, true, "1"},
	} {
		t.Run(test.name, func(t *testing.T) {
			environment := map[string]string{"LLAMA_ARG_GPU_VRAM_LIMIT_MIB": "15360", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
			setProjectorLifecycleEnvironment(environment, test.projectors, test.supported)
			if environment[projectorOnDemandEnvironment] != test.want || environment["LLAMA_ARG_GPU_VRAM_LIMIT_MIB"] != "15360" || environment["OLLAMA_KV_CACHE_TYPE"] != "q8_0" {
				t.Fatalf("lifecycle changed unrelated model policies: %v", environment)
			}
			cmd := &exec.Cmd{}
			SetupLlamaServerCommandEnv(cmd, "llama-server", nil, environment)
			var found bool
			for _, entry := range cmd.Env {
				key, value, _ := strings.Cut(entry, "=")
				if key == projectorOnDemandEnvironment {
					found = true
					if value != test.want || value == "" {
						t.Fatalf("unproved lifecycle policy passed to native: %q", entry)
					}
				}
			}
			if found != (test.want != "") || os.Getenv(projectorOnDemandEnvironment) != "1" {
				t.Fatal("fallback did not clear inheritance or changed server environment")
			}
		})
	}
}

func TestAdaptivePrefixCacheRespectsPositiveAllowanceAndExplicitClear(t *testing.T) {
	t.Setenv("LLAMA_ARG_CACHE_RAM", "8192")
	for _, test := range []struct {
		name     string
		value    *string
		decision bool
		want     bool
	}{
		{"inherit configured cache", nil, false, true},
		{"explicit zero", new("0"), false, false},
		{"explicit clear", new(""), false, false},
		{"explicit positive", new("1024"), false, true},
		{"invalid negative", new("-1"), false, false},
		{"invalid value", new("invalid"), false, false},
		{"decision cache disabled", nil, true, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			opts := api.DefaultOptions()
			if test.value != nil {
				opts.RunnerEnv = map[string]string{"LLAMA_ARG_CACHE_RAM": *test.value}
			}
			requested := adaptivePrefixCacheRequested(opts, test.decision)
			if requested != test.want {
				t.Fatalf("adaptive cache request=%v, want=%v", requested, test.want)
			}
			for _, supported := range []bool{false, true} {
				environment := map[string]string{"LLAMA_ARG_CACHE_RAM": "8192"}
				setAdaptivePrefixCacheEnvironment(environment, requested, supported)
				want := ""
				if requested && supported {
					want = "1"
				}
				if environment[adaptivePrefixCacheEnvironment] != want || environment["LLAMA_ARG_CACHE_RAM"] != "8192" {
					t.Fatalf("adaptive cache changed the configured ceiling: %v", environment)
				}
			}
		})
	}
}
