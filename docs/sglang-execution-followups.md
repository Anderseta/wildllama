# Execution follow-ups: overlap, residual fusion and speculation

This update implements bounded follow-ups to the existing SGLang-inspired
execution pipeline. It does not replace the GGML engine with SGLang or claim
equal performance across hardware and models.

* [Response overlap](overlap-entropy-transactions.md) now accepts complete
  compatible stochastic backend chains, history penalties and grammar-first
  masks. A fixed host entropy checkpoint makes optional-forward rollback
  transactional without cloning vocabulary, grammar or history state.
* [Residual normalization fusion](residual-normalization.md) combines compatible
  CUDA ADD/RMSNorm/MUL operations, preserving the residual output and using its
  existing allocations. It uses the existing explicit dense experimental switch.
* [Speculative input staging](speculative-staging.md) reduces EAGLE3 host
  concatenation workspace, reuses MTP bookkeeping and avoids materializing unused
  capacity from an oversized MTP device bank.

## Memory contracts

The managed CUDA allocation quota continues to govern the allocations it tracks.
It is not the device's total memory consumption: driver/library allocations and
other programs are outside that quota. The MoE RAM weight allowance continues
to cover the selective locked weight cache, not process RSS or all runtime state.
No change enlarges those configured limits or reduces the requested context.

The new runner environment `WILDLLAMA_MTP_DEVICE_STATE_MIB` is an additional
per-context allowance, from 0 through 32 MiB, for the already optional
`WILDLLAMA_MTP_DEVICE_STATE=1` transport. Target and draft can have independent
contexts. Actual aligned transport banks count inside the managed CUDA quota;
this allowance never adds capacity to that quota. Insufficient optional space
uses the original host transport. Zero also selects host transport.

No existing saved profile is silently changed. Wildgoose exposes the transport
checkbox and budget in the model profile's additional-VRAM group, persists both
in its existing environment settings and includes a conservative two-context
allowance in the VRAM estimate. A change takes effect on runner reload.

## Validation boundary

Only short CPU sampling/operator-policy and small synthetic MTP contracts are
required for this iteration, together with type/configuration tests and native
compilation. CUDA runtime correctness, end-to-end throughput and trained-model
quality comparisons remain deferred while the user's GPU is occupied. The
experimental overlap, CUDA dense fusion and MTP transport stay explicit opt-ins;
the conservative defaults and already-proven optimizations are preserved.

The [validation record](benchmarks/execution-followups-20261101.json) separates
the five short CPU contracts, 384 discarded-forward token comparisons, Go
environment checks and 114 Wildgoose tests from the deferred CUDA/model checks.
The complete ordered 60-patch stack reproduces all 193 native files it owns.

Related SGLang designs: [overlap scheduling](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/overlap_utils.py),
[speculative decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding),
and [adaptive speculative decoding](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding).
This implementation adapts execution principles to the existing native engine;
it does not copy SGLang kernels or introduce its Python/CUDA dependencies.
