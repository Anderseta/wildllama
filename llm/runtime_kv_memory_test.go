package llm

import (
	"encoding/json"
	"io"
	"strings"
	"sync"
	"testing"
)

func kvRuntimeFixture() (*llamaServerRunner, *memoryParsingWriter) {
	runner := &llamaServerRunner{}
	writer := &memoryParsingWriter{runner: runner, inner: io.Discard}
	runner.output = writer
	return runner, writer
}

func TestRuntimeKVCommitmentDistinguishesLegacyInitialAndMeasuredCurrent(t *testing.T) {
	runner, writer := kvRuntimeFixture()
	_, _ = writer.Write([]byte("load_tensors: offloaded 66/66 layers to GPU\nllama_kv_cache: CUDA0 KV buffer size = 320.00 MiB\nKV growth: 8192 initial cells, 65536 logical cells, 2176.00 MiB virtual capacity\n"))
	got := runner.RuntimeMemory()
	if got.KVCommitted != nil || got.GPU.KVBytes != 320*1024*1024 || got.KVCapacityBytes != 2176*1024*1024 {
		t.Fatalf("startup bytes must not imply current commitment: %+v", got)
	}
	encoded, err := json.Marshal(got)
	if err != nil || strings.Contains(string(encoded), "kv_committed") {
		t.Fatalf("unknown commitment must be omitted: %s, %v", encoded, err)
	}
	line := "wildllama_activity {\"kind\":\"kv_memory\",\"scope\":\"target\",\"gpu_bytes\":1073741824,\"host_bytes\":0,\"capacity_bytes\":2281701376}\n"
	for index := range len(line) {
		_, _ = writer.Write([]byte(line[index : index+1]))
	}
	got = runner.RuntimeMemory()
	if got.KVCommitted == nil || got.KVCommitted.GPUBytes != 1024*1024*1024 || got.KVCommitted.HostBytes != 0 || got.KVCommitted.ObservedAt <= 0 || got.KVCommitted.Source != "native_context" {
		t.Fatalf("missing native commitment: %+v", got)
	}
	if got.GPU.KVBytes != 320*1024*1024 {
		t.Fatal("legacy load-time fields must retain their original meaning")
	}
	got.KVCommitted.GPUBytes = 1
	if runner.RuntimeMemory().KVCommitted.GPUBytes != 1024*1024*1024 {
		t.Fatal("caller mutated a shared native commitment")
	}
	_, _ = writer.Write([]byte("llama_context: constructing llama_context\n"))
	if runner.RuntimeMemory().KVCommitted != nil {
		t.Fatal("replacement context inherited old live commitment")
	}
}

func TestRuntimeKVCommitmentRequiresACompleteTargetCounter(t *testing.T) {
	for _, line := range []string{
		`private prompt: wildllama_activity {"kind":"kv_memory","scope":"target","gpu_bytes":1,"host_bytes":0,"capacity_bytes":2}`,
		`wildllama_activity {"kind":"kv_memory","scope":"draft","gpu_bytes":1,"host_bytes":0,"capacity_bytes":2}`,
		`wildllama_activity {"kind":"kv_memory","scope":"target","gpu_bytes":1,"capacity_bytes":2}`,
		`wildllama_activity {"kind":"kv_memory","scope":"target","gpu_bytes":-1,"host_bytes":0,"capacity_bytes":2}`,
		`wildllama_activity {"kind":"kv_memory","scope":"target","gpu_bytes":18446744073709551615,"host_bytes":1,"capacity_bytes":2}`,
		`wildllama_activity {"kind":"memory","stage":"cuda","memory":{"vram_bytes":123}}`,
	} {
		t.Run(line, func(t *testing.T) {
			runner, writer := kvRuntimeFixture()
			_, _ = writer.Write([]byte(line + "\n"))
			if runner.RuntimeMemory().KVCommitted != nil {
				t.Fatal("partial or unrelated counter became a physical measurement")
			}
		})
	}
	runner, writer := kvRuntimeFixture()
	_, _ = writer.Write([]byte("wildllama_activity {\"kind\":\"kv_memory\",\"scope\":\"target\",\"gpu_bytes\":0,\"host_bytes\":0,\"capacity_bytes\":0}\n"))
	if runner.RuntimeMemory().KVCommitted == nil || runner.RuntimeMemory().KVCommitted.GPUBytes != 0 {
		t.Fatal("explicit measured zero must remain distinguishable from unknown")
	}
}

func TestRuntimeKVCommitmentConcurrentUpdatesAndHostPlacement(t *testing.T) {
	runner, writer := kvRuntimeFixture()
	_, _ = writer.Write([]byte("load_tensors: offloaded 66/66 layers to GPU\n"))
	var group sync.WaitGroup
	group.Add(2)
	go func() {
		defer group.Done()
		for range 100 {
			_, _ = writer.Write([]byte("wildllama_activity {\"kind\":\"kv_memory\",\"scope\":\"target\",\"gpu_bytes\":20,\"host_bytes\":10,\"capacity_bytes\":32}\n"))
		}
	}()
	go func() {
		defer group.Done()
		for range 100 {
			_ = runner.RuntimeMemory()
		}
	}()
	group.Wait()
	if got := runner.RuntimeMemory(); got.Placement != "offload" || got.KVCommitted == nil || got.KVCommitted.HostBytes != 10 {
		t.Fatalf("live host cache must count as offload: %+v", got)
	}
}
