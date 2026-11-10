package llm

import (
	"fmt"
	"testing"
)

func TestRuntimeProjectorMemoryTracksLifecycleWithoutResettingTargetKV(t *testing.T) {
	runner, writer := kvRuntimeFixture()
	_, _ = writer.Write([]byte("load_tensors: offloaded 66/66 layers to GPU\nllama_kv_cache: CUDA0 KV buffer size = 320.00 MiB\nsched_reserve: CUDA0 compute buffer size = 420.00 MiB\n"))
	_, _ = writer.Write([]byte("wildllama_activity {\"kind\":\"kv_memory\",\"scope\":\"target\",\"gpu_bytes\":1000,\"host_bytes\":0,\"capacity_bytes\":2000}\n"))
	_, _ = writer.Write([]byte("wildllama_mmproj state=metadata\nwildllama_mmproj state=loading progress=0.25\nclip_model_loader: CUDA0 model buffer size = 888.00 MiB\nreserve_compute_meta: CUDA0 compute buffer size = 248.00 MiB\nreserve_compute_meta: CPU compute buffer size = 24.00 MiB\nwildllama_mmproj state=loading progress=0.9\nwildllama_mmproj state=loaded\n"))
	got := runner.RuntimeMemory()
	if got.ProjectorState != "loaded" || got.GPU.ComputeBytes != 668*1024*1024 || got.Host.ComputeBytes != 24*1024*1024 || got.KVCommitted == nil || got.GPU.KVBytes != 320*1024*1024 {
		t.Fatalf("projector load corrupted target accounting: %+v", got)
	}
	_, _ = writer.Write([]byte("wildllama_mmproj state=unloaded\n"))
	got = runner.RuntimeMemory()
	if got.ProjectorState != "unloaded" || got.GPU.ComputeBytes != 420*1024*1024 || got.Host.ComputeBytes != 0 || got.GPU.WeightsBytes != 0 || got.KVCommitted == nil {
		t.Fatalf("unloaded projector retained buffers or discarded target KV: %+v", got)
	}
	for cycle := range 2 {
		_, _ = writer.Write([]byte("wildllama_mmproj state=loading progress=0\nreserve_compute_meta: CUDA0 compute buffer size = 248.00 MiB\nwildllama_mmproj state=loaded\n"))
		if got := runner.RuntimeMemory(); got.GPU.ComputeBytes != 668*1024*1024 {
			t.Fatalf("cycle %d duplicated clip buffers: %+v", cycle, got)
		}
		_, _ = writer.Write([]byte("wildllama_mmproj state=unloaded\n"))
	}
	runner.resetLoadAccounting()
	if got := runner.RuntimeMemory(); got.ProjectorState != "" || got.ProjectorProgress != nil || got.KVCommitted != nil {
		t.Fatalf("replacement process retained old observations: %+v", got)
	}
}

func TestRuntimeProjectorMemoryProgressIsBoundedAndPrivateTextIsIgnored(t *testing.T) {
	for _, progress := range []string{"NaN", "Inf", "-0.1", "1.1", "private"} {
		runner, writer := kvRuntimeFixture()
		_, _ = writer.Write([]byte(fmt.Sprintf("wildllama_mmproj state=loading progress=%s\n", progress)))
		if got := runner.RuntimeMemory(); got.ProjectorState != "loading" || got.ProjectorProgress != nil {
			t.Fatalf("invalid progress %q: %+v", progress, got)
		}
	}
	runner, writer := kvRuntimeFixture()
	_, _ = writer.Write([]byte("private prompt: wildllama_mmproj state=loaded\n"))
	if runner.RuntimeMemory().ProjectorState != "" {
		t.Fatal("user text became native projector state")
	}
	line := "wildllama_mmproj state=loading progress=0.5\n"
	for index := range len(line) {
		_, _ = writer.Write([]byte(line[index : index+1]))
	}
	if got := runner.RuntimeMemory(); got.ProjectorProgress == nil || *got.ProjectorProgress != 0.5 {
		t.Fatalf("fragmented native progress lost: %+v", got)
	}
}
