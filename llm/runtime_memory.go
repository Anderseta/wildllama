package llm

import "github.com/ollama/ollama/api"

// RuntimeMemoryReporter is optional; other backends must not invent placement
// from size == size_vram, which can also be an unobserved estimate.
type RuntimeMemoryReporter interface {
	RuntimeMemory() api.RuntimeMemory
}

func (s *llamaServerRunner) RuntimeMemory() api.RuntimeMemory {
	managed, observed := s.status.managedVRAM()
	s.memoryMu.RLock()
	defer s.memoryMu.RUnlock()
	initial, _ := api.ResolveKVCacheInitialTokens(s.options)
	projectorCPU := false
	if len(s.launch.projectors) > 0 {
		projectorCPU, _ = s.launch.mmprojOffloadDisabled()
	}
	result := api.RuntimeMemory{
		Placement: "unknown", LayersObserved: s.layerPlacementObserved,
		GPULayers: s.gpuLayers, TotalLayers: s.totalLayers, OverflowLayers: s.gpuLayerOverflow,
		WeightStreaming:    s.moeRAMWeightLimit > 0,
		AllocatedVRAMBytes: s.memGPU, AllocationSource: "native_logs",
		ContextLength: s.options.NumCtx, KVInitialTokens: min(initial, max(0, s.options.NumCtx)),
		ProjectorOffloadCPU: projectorCPU,
	}
	if observed {
		result.AllocatedVRAMBytes, result.AllocationSource = managed, "cuda_allocator"
	}
	if s.output != nil {
		result.WeightStreaming = result.WeightStreaming || s.output.boundedExpertCache > 0
		for key, value := range s.output.buffers {
			buffers := &result.Host
			if isGPUBuffer(key.backend) {
				buffers = &result.GPU
			}
			switch key.kind {
			case "model":
				buffers.WeightsBytes += value.bytes
			case "KV":
				buffers.KVBytes += value.bytes
				result.KVCapacityBytes += max(value.bytes, value.capacityBytes)
				result.KVGrowthObserved = result.KVGrowthObserved || value.growing
			case "RS":
				buffers.RecurrentBytes += value.bytes
			case "compute":
				buffers.ComputeBytes += value.bytes
			case "output":
				buffers.OutputBytes += value.bytes
			}
		}
	}
	// Host mappings for embeddings and pinned I/O exist with full GPU execution.
	// They are not evidence of layer offload, nor are they resident process RAM.
	if result.LayersObserved && result.TotalLayers > 0 && result.GPULayers == 0 {
		result.Placement = "cpu"
	} else if result.WeightStreaming || result.OverflowLayers > 0 || result.Host.KVBytes > 0 || result.Host.RecurrentBytes > 0 || result.ProjectorOffloadCPU {
		result.Placement = "offload"
	} else if result.LayersObserved && result.TotalLayers > 0 {
		switch {
		case result.GPULayers < result.TotalLayers:
			result.Placement = "offload"
		default:
			result.Placement = "gpu"
		}
	}
	return result
}
