package cudaload

import "strings"

const ModuleLoadingEnv = "CUDA_MODULE_LOADING"

// IsSharedObjectInitFailure recognizes CUDA error 303. A later specific CUDA
// error supersedes it; a bare source-location line does not.
func IsSharedObjectInitFailure(message string) bool {
	matched := false
	for _, line := range strings.Split(strings.ToLower(message), "\n") {
		if strings.Contains(line, "cudaerrorsharedobjectinitfailed") ||
			(strings.Contains(line, "cuda error") && strings.Contains(line, "shared object initialization failed")) {
			matched = true
		} else if strings.Contains(line, "cuda error:") {
			matched = false
		}
	}
	return matched
}

func HasExplicitModuleLoading(env []string) bool {
	for _, entry := range env {
		key, value, ok := strings.Cut(entry, "=")
		if ok && strings.EqualFold(key, ModuleLoadingEnv) && strings.TrimSpace(value) != "" {
			return true
		}
	}
	return false
}
