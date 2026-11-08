package llm

import (
	"io"
	"sync"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestRuntimeMemoryPlacementEvidence(t *testing.T) {
	for _, tc := range []struct{ name, logs, want string }{
		{"unknown is not full GPU", "load_tensors: CUDA0 model buffer size = 100.00 MiB\n", "unknown"},
		{"mmap is not CPU execution", "load_tensors: offloaded 66/66 layers to GPU\nload_tensors: CPU_Mapped model buffer size = 388.38 MiB\nsched_reserve: CUDA_Host compute buffer size = 40.54 MiB\n", "gpu"},
		{"partial layers", "load_tensors: offloaded 54/66 layers to GPU\n", "offload"},
		{"CPU only", "load_tensors: offloaded 0/66 layers to GPU\n", "cpu"},
		{"host KV overrides full layers", "load_tensors: offloaded 66/66 layers to GPU\nllama_kv_cache: CPU KV buffer size = 12.00 MiB\n", "offload"},
		{"host recurrence overrides full layers", "load_tensors: offloaded 66/66 layers to GPU\nllama_memory_recurrent: CPU RS buffer size = 12.00 MiB\n", "offload"},
		{"partial layer overflow", "common_params_fit_impl:   - CUDA0 (GPU): 66 layers ( 2 overflowing), 100 MiB used, 200 MiB free\nload_tensors: offloaded 66/66 layers to GPU\n", "offload"},
		{"expert streaming", "load_tensors: offloaded 66/66 layers to GPU\ngpu-exclusive: 100.00 MiB bounded expert cache\n", "offload"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runner := &llamaServerRunner{}
			w := &memoryParsingWriter{runner: runner, inner: io.Discard}
			runner.output = w
			// Real pipe output can split a number or buffer line anywhere.
			for i := range len(tc.logs) {
				_, _ = w.Write([]byte(tc.logs[i : i+1]))
			}
			got := runner.RuntimeMemory()
			if got.Placement != tc.want {
				t.Fatalf("placement = %s, want %s: %+v", got.Placement, tc.want, got)
			}
		})
	}
}

func TestRuntimeMemorySeparatesLiveAllocationFromStartupBuffers(t *testing.T) {
	const mib = uint64(1024 * 1024)
	initial := 8192
	opts := api.DefaultOptions()
	opts.NumCtx, opts.KVCacheInitialTokens = 32768, &initial
	status := NewStatusWriter(io.Discard)
	runner := &llamaServerRunner{options: opts, status: status}
	w := &memoryParsingWriter{runner: runner, inner: status}
	runner.output = w
	_, _ = w.Write([]byte("load_tensors: offloaded 66/66 layers to GPU\nllama_context: constructing llama_context\nllama_kv_cache: CUDA0 KV buffer size = 320.00 MiB\nKV growth: 8192 initial cells, 32768 logical cells, 1088.00 MiB virtual capacity\nllama_memory_recurrent: CUDA0 RS buffer size = 149.62 MiB\n"))
	_, _ = status.Write([]byte("wildllama_activity {\"kind\":\"memory\",\"stage\":\"cuda\",\"memory\":{\"vram_bytes\":999999999}}\n"))
	got := runner.RuntimeMemory()
	if got.GPU.KVBytes != 320*mib || got.AllocatedVRAMBytes != 999999999 || got.AllocationSource != "cuda_allocator" || got.KVInitialTokens != 8192 || got.ContextLength != 32768 {
		t.Fatalf("unexpected snapshot: %+v", got)
	}
	if got.KVCapacityBytes != 1088*mib || !got.KVGrowthObserved {
		t.Fatalf("initial commitment was confused with logical KV capacity: %+v", got)
	}
	// Polling status must not race the incremental log parser.
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for range 100 {
			_, _ = w.Write([]byte("sched_reserve: CUDA0 compute buffer size = 260.54 MiB\n"))
		}
	}()
	go func() {
		defer wg.Done()
		for range 100 {
			_ = runner.RuntimeMemory()
		}
	}()
	wg.Wait()
	runner.resetLoadAccounting()
	if runner.RuntimeMemory().LayersObserved {
		t.Fatal("a replacement runner inherited confirmed layers")
	}
}

func TestRuntimeMemoryReplacesProbeContextPlacement(t *testing.T) {
	runner := &llamaServerRunner{}
	w := &memoryParsingWriter{runner: runner, inner: io.Discard}
	runner.output = w
	_, _ = w.Write([]byte("load_tensors: offloaded 54/66 layers to GPU\nllama_kv_cache: CPU KV buffer size = 612.00 MiB\nllama_memory_recurrent: CPU RS buffer size = 28.05 MiB\n"))
	_, _ = w.Write([]byte("load_tensors: offloaded 66/66 layers to GPU\nllama_context: constructing llama_context\nllama_kv_cache: CUDA0 KV buffer size = 320.00 MiB\nKV growth: 8192 initial cells, 32768 logical cells, 1088.00 MiB virtual capacity\nllama_memory_recurrent: CUDA0 RS buffer size = 149.62 MiB\n"))
	got := runner.RuntimeMemory()
	if got.Placement != "gpu" || got.Host.KVBytes != 0 || got.Host.RecurrentBytes != 0 || got.KVCapacityBytes != 1088*1024*1024 {
		t.Fatalf("a probe leaked into the confirmed placement: %+v", got)
	}
}
