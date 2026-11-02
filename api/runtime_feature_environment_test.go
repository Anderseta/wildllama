package api

import "testing"

func TestRuntimeFeatureRunnerEnvironment(t *testing.T) {
	keys := []string{
		"WILDLLAMA_DRY_BACKEND", "WILDLLAMA_DECODE_OVERLAP", "WILDLLAMA_MTP_DEVICE_STATE", "WILDLLAMA_KV_CACHE_PAGING",
	}
	for _, key := range keys {
		t.Run(key, func(t *testing.T) {
			for _, value := range []string{"", "0", "1"} {
				input := map[string]string{key: value}
				parsed, err := ParseRunnerEnvironment(input)
				if err != nil || parsed[key] != value {
					t.Fatalf("profile setting %q: parsed=%v err=%v", value, parsed, err)
				}
				input[key] = "changed"
				if parsed[key] != value {
					t.Fatal("runner options retained the caller's mutable map")
				}
			}
			for _, value := range []string{"true", "auto", "2", "-1", "1\nPATH=other"} {
				if _, err := ParseRunnerEnvironment(map[string]string{key: value}); err == nil {
					t.Fatalf("accepted invalid profile value %q", value)
				}
			}
		})
	}
	for _, value := range []string{"", "0", "128", "1024"} {
		if _, err := ParseRunnerEnvironment(map[string]string{"WILDLLAMA_GRAPH_VARIANTS_MIB": value}); err != nil {
			t.Fatalf("graph cache value %q: %v", value, err)
		}
	}
	for _, value := range []string{"-1", "1025", "1.5", "128\n"} {
		if _, err := ParseRunnerEnvironment(map[string]string{"WILDLLAMA_GRAPH_VARIANTS_MIB": value}); err == nil {
			t.Fatalf("accepted invalid graph budget %q", value)
		}
	}
	for _, value := range []string{"", "0", "1", "32"} {
		if _, err := ParseRunnerEnvironment(map[string]string{"WILDLLAMA_MTP_DEVICE_STATE_MIB": value}); err != nil {
			t.Fatalf("MTP state budget %q: %v", value, err)
		}
	}
	for _, value := range []string{"-1", "33", "1.5", "32\n"} {
		if _, err := ParseRunnerEnvironment(map[string]string{"WILDLLAMA_MTP_DEVICE_STATE_MIB": value}); err == nil {
			t.Fatalf("accepted invalid MTP state budget %q", value)
		}
	}
	for _, key := range []string{"WILDLLAMA_PREFILL_TRACE", "WILDLLAMA_PREFILL_LATENCY_US", "PATH", "HF_TOKEN"} {
		if _, err := ParseRunnerEnvironment(map[string]string{key: "1"}); err == nil {
			t.Fatalf("profile runtime controls admitted unrelated or diagnostic key %q", key)
		}
	}
	for _, key := range []string{"WILDLLAMA_VERIFY_DEVICE_MIB", "WILDLLAMA_PREFIX_DISK_PAGE_CACHE_MIB"} {
		for _, value := range []string{"", "0", "1", "64"} {
			if _, err := ParseRunnerEnvironment(map[string]string{key: value}); err != nil {
				t.Fatalf("%s budget %q: %v", key, value, err)
			}
		}
		for _, value := range []string{"-1", "+1", "-0", "65", "1.5", "64\n", "99999999999999999999"} {
			if _, err := ParseRunnerEnvironment(map[string]string{key: value}); err == nil {
				t.Fatalf("accepted invalid %s budget %q", key, value)
			}
		}
	}
}
