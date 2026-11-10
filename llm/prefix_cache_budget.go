package llm

import (
	"strconv"

	"github.com/ollama/ollama/api"
)

const adaptivePrefixCacheEnvironment = "WILDLLAMA_PREFIX_RAM_ADAPTIVE"

func adaptivePrefixCacheRequested(opts api.Options, decision bool) bool {
	if decision {
		// The decision runner explicitly disables its native prefix cache.
		return false
	}
	value, err := strconv.ParseUint(gpuNativeEnvironment(opts, "LLAMA_ARG_CACHE_RAM"), 10, 64)
	return err == nil && value > 0
}

func setAdaptivePrefixCacheEnvironment(environment map[string]string, requested, supported bool) {
	// Native adaptation can only reduce a configured positive cache allowance;
	// it must neither enable caching nor contradict an explicit API clear/zero.
	environment[adaptivePrefixCacheEnvironment] = ""
	if requested && supported {
		environment[adaptivePrefixCacheEnvironment] = "1"
	}
}
