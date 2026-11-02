# Bounded adaptive prompt slices

The existing explicit `WILDLLAMA_ADAPTIVE_PREFILL=1` experiment can now admit a
measured prompt slice beside an already-generating sequence, instead of only
alternating pure generation with a full prompt round. The feature stays off
unless explicitly selected. No saved model profile, batch/microbatch setting,
logical context, sampling rule or weight precision is changed.

The controller learns one fixed-point cost per prompt row from completed mixed
decode calls after subtracting the pure-generation baseline for the same number
of generating rows. Admission still requires two pure observations and sustained
mixed latency pressure. It uses the existing explicit latency objective, or the
existing relative objective of twice baseline latency with a 1 ms floor.

An affordable slice can run every round. Slice sizes are bounded by the existing
batch capacity, use powers of two to avoid shape churn, and retain a throughput
floor of at most 32 prompt rows. The last slice can be shorter; no padding or
additional tokens are evaluated. If that floor cannot meet the objective, the
original alternating-deferral fairness rule remains: a deferred round is always
followed by an admitted prompt round. The admitted round now uses the bounded
slice. The objective is guidance, not a real-time latency guarantee.

This is still restricted to eligible ordinary completion work with supported
plain unified KV storage and concurrent generation. Hybrid/recurrent memories,
SWA, speculative/MTP execution, media, LoRA, constrained grammars and managed
weight-streaming providers retain their previous admission. A change in active
generation rows or compute reservation invalidates the relevant learned cost.
Isolated prompts and unsupported requests retain the ordinary prompt limit.

The implementation adds one scalar cost estimate and one counter to the bounded
CPU controller. It introduces no GPU tensor, workspace, synchronization, retained
arena or new memory allowance. The existing reuse-only contention scope and
reservation-growth guard also protect sliced rounds; existing RAM/VRAM scopes
are preserved. It is not a process-RSS or device-total memory limiter.

The short CPU policy tests cover measured slices, continuous prompt progress,
unreachable targets, generation/reservation changes, throughput floors and
integer bounds. The existing tiny inference fixture additionally executes mixed
32/64/128-row slices under the reuse-only scope and compares its logits with the
same ordinary batch, checking secondary memory admission. These are correctness
and resource contracts, not trained-model throughput or SGLang parity results.
CUDA behavior and end-to-end throughput require separate measurements before
changing defaults.
