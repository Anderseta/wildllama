package llm

import (
	"bytes"
	"encoding/json"
	"time"

	"github.com/ollama/ollama/api"
)

// observeKVCommitmentLocked accepts only the framed native target-context
// event. Explicit zero counters are valid; absent counters are not observations.
// The caller holds runner.memoryMu, including during reset and snapshot reads.
func (w *memoryParsingWriter) observeKVCommitmentLocked(line []byte) bool {
	const marker = "wildllama_activity "
	if !bytes.HasPrefix(line, []byte(marker)) {
		return false
	}
	var event struct {
		Kind          string  `json:"kind"`
		Scope         string  `json:"scope"`
		GPUBytes      *uint64 `json:"gpu_bytes"`
		HostBytes     *uint64 `json:"host_bytes"`
		CapacityBytes *uint64 `json:"capacity_bytes"`
	}
	if json.Unmarshal(line[len(marker):], &event) != nil || event.Kind != "kv_memory" {
		return false
	}
	if event.Scope != "target" || event.GPUBytes == nil || event.HostBytes == nil || event.CapacityBytes == nil {
		return true
	}
	gpu, host, capacity := *event.GPUBytes, *event.HostBytes, *event.CapacityBytes
	if gpu > ^uint64(0)-host {
		return true
	}
	w.kvCommitted = &api.RuntimeKVCommitment{
		GPUBytes: gpu, HostBytes: host, CapacityBytes: capacity,
		ObservedAt: time.Now().UnixMilli(), Source: "native_context",
	}
	return true
}
