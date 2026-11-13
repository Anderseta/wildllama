package imagegen

// MiniMax exposes layer boundaries to the native allocator. Let it reduce
// workspace residency before restarting the entire reference-encoding pipeline
// in RAM. Other architectures retain their validated placement policy.
func allowMiniMaxSegmentation(p Profile, mode imageMemoryMode) bool {
	if mode != imageGPUResident || !p.StageComponents || !p.OffloadToCPU {
		return false
	}
	adapter := detectImageReferenceAdapter(p.DiffusionModel)
	return adapter != nil && adapter.family == "minimax-h3-image"
}

// Attention heads are independent. Grouping them bounds temporary Q/K storage
// without dropping tokens or changing precision. Large GPUs keep the wider path.
func miniMaxAttentionArguments(p Profile, r Request) []string {
	if len(r.ReferenceImages) < 2 || p.MaxVRAMMiB > 16*1024 {
		return nil
	}
	return []string{"--model-args", "minimax_h3_attn_chunk_budget=0.03125"}
}
