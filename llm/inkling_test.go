package llm

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/ml"
)

func TestInklingFitAndCache(t *testing.T) {
	launch := llamaServerLaunchConfig{
		modelArch: inkling2Architecture, kvCacheType: "q8_0",
		projectors: []string{"vision"}, mmprojMemory: 600 << 20,
		gpus: []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA"}, ComputeMajor: 12, FreeMemory: 15 << 30}},
		opts: api.DefaultOptions(),
	}
	launch.opts.RunnerEnv = map[string]string{
		"OLLAMA_FLASH_ATTENTION": "1", "WILDLLAMA_FP8_KV": "1",
		"LLAMA_ARG_CACHE_TYPE_K": "", "LLAMA_ARG_CACHE_TYPE_V": "",
	}
	if _, pad := launch.mmprojFitTargetMiB(); pad {
		t.Fatal("engine already includes the projector in fit")
	}
	check := func(k, v string) {
		t.Helper()
		if gotK, gotV := launch.cacheStorageTypes(); gotK != k || gotV != v {
			t.Fatalf("cache (%s,%s), want (%s,%s)", gotK, gotV, k, v)
		}
	}
	check("f8_e4m3", "f8_e4m3")
	launch.opts.RunnerEnv["LLAMA_ARG_CACHE_TYPE_K"] = "f16"
	check("f16", "f8_e4m3")
	launch.opts.RunnerEnv["WILDLLAMA_FP8_KV"] = "0"
	check("f16", "q8_0")
	launch.opts.RunnerEnv["LLAMA_ARG_CACHE_TYPE_K"] = ""
	launch.opts.RunnerEnv["WILDLLAMA_FP8_KV"] = "1"
	launch.opts.RunnerEnv["OLLAMA_FLASH_ATTENTION"] = "0"
	check("q8_0", "q8_0")
	launch.opts.RunnerEnv["OLLAMA_FLASH_ATTENTION"] = "1"
	launch.modelArch = "qwen35"
	check("q8_0", "q8_0")
	if _, pad := launch.mmprojFitTargetMiB(); !pad {
		t.Fatal("missing projector allowance for the primary runner")
	}
}
