package llm

import (
	"os"
	"os/exec"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestRunnerEnvironmentClearDoesNotPassEmptyNativeArgument(t *testing.T) {
	t.Setenv("LLAMA_ARG_CACHE_RAM", "4096")
	cmd := &exec.Cmd{}
	SetupLlamaServerCommandEnv(cmd, "llama-server", nil, map[string]string{
		"LLAMA_ARG_CACHE_RAM": "", "LLAMA_ARG_MOE_RAM_LIMIT_MIB": "", "CUDA_VISIBLE_DEVICES": "",
	})
	foundCUDA := false
	for _, entry := range cmd.Env {
		key, value, _ := strings.Cut(entry, "=")
		if key == "LLAMA_ARG_CACHE_RAM" || key == "LLAMA_ARG_MOE_RAM_LIMIT_MIB" {
			t.Fatalf("empty override reached native argument parser: %q", entry)
		}
		if key == "CUDA_VISIBLE_DEVICES" && value == "" {
			foundCUDA = true
		}
	}
	if !foundCUDA || os.Getenv("LLAMA_ARG_CACHE_RAM") != "4096" {
		t.Fatal("GPU visibility or parent environment changed")
	}
}

func TestMTPTransportEnvironmentPreservesMemoryPolicies(t *testing.T) {
	t.Setenv("WILDLLAMA_MTP_DEVICE_STATE_MIB", "4")
	cmd := &exec.Cmd{}
	SetupLlamaServerCommandEnv(cmd, "llama-server", nil, map[string]string{
		"WILDLLAMA_MTP_DEVICE_STATE": "1", "WILDLLAMA_MTP_DEVICE_STATE_MIB": "",
		"LLAMA_ARG_GPU_VRAM_LIMIT_MIB": "12288", "LLAMA_ARG_MOE_RAM_LIMIT_MIB": "10240",
	})
	policies := map[string]string{}
	for _, entry := range cmd.Env {
		key, value, _ := strings.Cut(entry, "=")
		if key == "WILDLLAMA_MTP_DEVICE_STATE_MIB" {
			t.Fatalf("cleared MTP allowance replaced the native default: %q", entry)
		}
		policies[key] = value
	}
	if policies["WILDLLAMA_MTP_DEVICE_STATE"] != "1" || policies["LLAMA_ARG_GPU_VRAM_LIMIT_MIB"] != "12288" || policies["LLAMA_ARG_MOE_RAM_LIMIT_MIB"] != "10240" {
		t.Fatalf("transport changed memory policies: %v", policies)
	}
	if os.Getenv("WILDLLAMA_MTP_DEVICE_STATE_MIB") != "4" {
		t.Fatal("runner override changed the parent process")
	}
}

func TestBoundedDeviceFeaturesPreserveRunnerPolicies(t *testing.T) {
	t.Setenv("WILDLLAMA_VERIFY_DEVICE_MIB", "32")
	t.Setenv("WILDLLAMA_PREFIX_DISK_PAGE_CACHE_MIB", "4")
	for _, budget := range []string{"", "0", "64"} {
		t.Run("budget="+budget, func(t *testing.T) {
			cmd := &exec.Cmd{}
			SetupLlamaServerCommandEnv(cmd, "llama-server", nil, map[string]string{
				"WILDLLAMA_VERIFY_DEVICE_MIB": budget, "WILDLLAMA_KV_CACHE_PAGING": "1",
				"WILDLLAMA_PREFIX_DISK_PAGE_CACHE_MIB": budget,
				"LLAMA_ARG_GPU_VRAM_LIMIT_MIB":         "12288", "LLAMA_ARG_MOE_RAM_LIMIT_MIB": "10240",
				"WILDLLAMA_KV_CACHE_INITIAL_TOKENS": "8192", "LLAMA_ARG_CTX_SIZE": "131072",
			})
			policies := map[string]string{}
			for _, entry := range cmd.Env {
				key, value, _ := strings.Cut(entry, "=")
				policies[key] = value
			}
			if value, present := policies["WILDLLAMA_VERIFY_DEVICE_MIB"]; present != (budget != "") || value != budget {
				t.Fatalf("explicit off/default budget changed: %q present=%v", value, present)
			}
			if value, present := policies["WILDLLAMA_PREFIX_DISK_PAGE_CACHE_MIB"]; present != (budget != "") || value != budget {
				t.Fatalf("explicit SSD metadata off/default budget changed: %q present=%v", value, present)
			}
			for key, value := range map[string]string{
				"WILDLLAMA_KV_CACHE_PAGING": "1", "LLAMA_ARG_GPU_VRAM_LIMIT_MIB": "12288",
				"LLAMA_ARG_MOE_RAM_LIMIT_MIB": "10240", "WILDLLAMA_KV_CACHE_INITIAL_TOKENS": "8192", "LLAMA_ARG_CTX_SIZE": "131072",
			} {
				if policies[key] != value {
					t.Fatalf("feature changed %s: %q", key, policies[key])
				}
			}
		})
	}
	if os.Getenv("WILDLLAMA_VERIFY_DEVICE_MIB") != "32" {
		t.Fatal("profile changed the parent process")
	}
	if os.Getenv("WILDLLAMA_PREFIX_DISK_PAGE_CACHE_MIB") != "4" {
		t.Fatal("profile changed the parent SSD metadata budget")
	}
}

func TestPagedAttentionEnvironmentPreservesRunnerPolicies(t *testing.T) {
	t.Setenv("WILDLLAMA_PAGED_ATTENTION", "1")
	for _, value := range []string{"", "0", "1"} {
		t.Run("profile="+value, func(t *testing.T) {
			input := map[string]string{
				"WILDLLAMA_PAGED_ATTENTION":    value,
				"WILDLLAMA_KV_CACHE_PAGING":    "0",
				"LLAMA_ARG_GPU_VRAM_LIMIT_MIB": "12288", "LLAMA_ARG_MOE_RAM_LIMIT_MIB": "10240",
				"WILDLLAMA_KV_CACHE_INITIAL_TOKENS": "8192", "LLAMA_ARG_CTX_SIZE": "131072",
				"LLAMA_ARG_CACHE_TYPE_K": "q8_0", "LLAMA_ARG_CACHE_TYPE_V": "q8_0",
			}
			validated, err := api.ParseRunnerEnvironment(map[string]string{"WILDLLAMA_PAGED_ATTENTION": value})
			if err != nil || validated["WILDLLAMA_PAGED_ATTENTION"] != value {
				t.Fatalf("invalid explicit paged attention setting: %v %v", validated, err)
			}
			cmd := &exec.Cmd{}
			SetupLlamaServerCommandEnv(cmd, "llama-server", nil, input)
			policies := map[string]string{}
			for _, entry := range cmd.Env {
				key, item, _ := strings.Cut(entry, "=")
				policies[key] = item
			}
			if actual, present := policies["WILDLLAMA_PAGED_ATTENTION"]; present != (value != "") || actual != value {
				t.Fatalf("explicit off/default paged attention changed: %q present=%v", actual, present)
			}
			for key, expected := range input {
				if key != "WILDLLAMA_PAGED_ATTENTION" && policies[key] != expected {
					t.Fatalf("paged attention changed %s: %q", key, policies[key])
				}
			}
		})
	}
	if os.Getenv("WILDLLAMA_PAGED_ATTENTION") != "1" {
		t.Fatal("runner opt-in changed the parent process")
	}
}

func TestLogicalAndPhysicalBatchAreIndependent(t *testing.T) {
	opts := api.DefaultOptions()
	opts.NumBatch = 2048
	opts.NumUbatch = 512
	got := appendBatchArgs(nil, opts, false, 1)
	if !reflect.DeepEqual(got, []string{"-b", "2048", "-ub", "512"}) {
		t.Fatalf("batch args %v", got)
	}
	_, allowance := LlamaServerBatch(loadTestGGUF(t, map[string]any{"general.architecture": "llama"}), opts, true)
	if allowance != 0 {
		t.Fatalf("logical batch incorrectly inflated physical scratch memory by %d", allowance)
	}
}

func TestExplicitSpeculationTypeOverridesDetectedDraft(t *testing.T) {
	opts := api.DefaultOptions()
	opts.SpeculativeType = "none"
	got := appendDraftArgs(nil, draftTypeMTP, "draft.gguf", opts)
	if !reflect.DeepEqual(got, []string{"--spec-type", "none"}) {
		t.Fatalf("draft not disabled: %v", got)
	}
	opts.SpeculativeType = "ngram-simple"
	got = appendDraftArgs(nil, draftTypeMTP, "draft.gguf", opts)
	if !reflect.DeepEqual(got, []string{"--spec-type", "ngram-simple", "--spec-draft-n-max", "4"}) {
		t.Fatalf("ngram should not load an extra model: %v", got)
	}
}
