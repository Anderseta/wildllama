package api

// RuntimeMemory is placement evidence from the loaded runner, not a prediction
// from file size or process RSS. Buffer sizes are startup/log snapshots; only
// AllocatedVRAMBytes and KVCommitted may come from live native measurements.
type RuntimeMemory struct {
	Placement           string               `json:"placement"`
	LayersObserved      bool                 `json:"layers_observed"`
	GPULayers           uint64               `json:"gpu_layers"`
	TotalLayers         uint64               `json:"total_layers"`
	OverflowLayers      int                  `json:"overflow_layers"`
	WeightStreaming     bool                 `json:"weight_streaming"`
	GPU                 RuntimeMemoryBuffers `json:"gpu"`
	Host                RuntimeMemoryBuffers `json:"host"`
	AllocatedVRAMBytes  uint64               `json:"allocated_vram_bytes"`
	AllocationSource    string               `json:"allocation_source"`
	ContextLength       int                  `json:"context_length"`
	KVInitialTokens     int                  `json:"kv_initial_tokens"`
	KVCapacityBytes     uint64               `json:"kv_capacity_bytes,omitempty"`
	KVGrowthObserved    bool                 `json:"kv_growth_observed"`
	KVCommitted         *RuntimeKVCommitment `json:"kv_committed,omitempty"`
	ProjectorState      string               `json:"projector_state,omitempty"`
	ProjectorProgress   *float64             `json:"projector_progress,omitempty"`
	ProjectorOffloadCPU bool                 `json:"projector_offload_cpu"`
}

// RuntimeKVCommitment is a live physical measurement for the target context.
// A nil pointer means the native payload did not publish this counter. Legacy
// GPU/Host.KVBytes remain load-time snapshots, never a substitute after growth.
type RuntimeKVCommitment struct {
	GPUBytes      uint64 `json:"gpu_bytes"`
	HostBytes     uint64 `json:"host_bytes"`
	CapacityBytes uint64 `json:"capacity_bytes"`
	ObservedAt    int64  `json:"observed_at"`
	Source        string `json:"source"`
}

type RuntimeMemoryBuffers struct {
	WeightsBytes   uint64 `json:"weights_bytes"`
	KVBytes        uint64 `json:"kv_bytes"`
	RecurrentBytes uint64 `json:"recurrent_bytes"`
	ComputeBytes   uint64 `json:"compute_bytes"`
	OutputBytes    uint64 `json:"output_bytes"`
}
