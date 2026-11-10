package api

import "testing"

func TestDenseCUDAOptionsRequireExplicitBoolean(t *testing.T) {
	for _, key := range []string{"GGML_CUDA_DENSE_OPT", "GGML_CUDA_IQ3_INPUT_REUSE", "WILDLLAMA_CUDA_GQA_Q8"} {
		for _, value := range []string{"", "0", "1"} {
			options, err := ParseRunnerEnvironment(map[string]string{key: value})
			if err != nil || options[key] != value {
				t.Fatalf("explicit comparison value lost for %s: %q %v", key, value, err)
			}
		}
		for _, value := range []string{"auto", "true", "-1", "2"} {
			if _, err := ParseRunnerEnvironment(map[string]string{key: value}); err == nil {
				t.Fatalf("unsupported dense mode accepted for %s: %q", key, value)
			}
		}
	}
}
