# MTP verification and recurrent-state efficiency

These changes optimize inference; they do not modify model weights, tokenizers,
context length or KV precision. Native support is advertised through
`--wildllama-runtime-features`. Wildgoose saves the switches per model and retains
explicit `0` values across restarts. Unsupported architectures and backends keep
their existing paths.

The deterministic block preview and extended compact rollback are enabled by
default on compatible paths; explicit `0` disables either feature. The other
switches below remain opt-in because they did not show consistent end-to-end
improvements in the local Huihui comparison. Existing sampling and MTP depth
settings are unchanged.

| Runner environment switch | Work affected | Additional device storage |
| --- | --- | --- |
| `WILDLLAMA_MTP_BLOCK_PREVIEW` | Reuse verifier scratch and preview deterministic filters without discarded random draws | None |
| `WILDLLAMA_CUDA_Q4_HEAD_SINGLE` | Specialized one-position Q4_K output projection | None; existing weights and buffers |
| `WILDLLAMA_MTP_PROPOSAL_TUNE` | Adapt the proposal temperature using observed target/proposal overlap | None; small per-sequence host state |
| `WILDLLAMA_MTP_DYNAMIC_VOCAB` | Compute the proposal head on a contextual shortlist | Bounded row indices and existing indexed-projection scratch; no second weight matrix |
| `WILDLLAMA_MTP_GPU_DISTRIBUTION` | Compute compatible filters and probabilities on CUDA | Optional sampler graph buffers; budget checks and raw-logit fallback |
| `WILDLLAMA_GDN_COMPACT_EXTENDED` | Compact recurrent rollback for three or four proposals | Reduces recurrent state storage |

## Sampling correctness

The deterministic preview retains filter ordering, penalties and reasoning
constraints. Grammar-last sampling is represented as its complete mixture:
the valid part of the initial distribution plus the probability of retrying
multiplied by the grammar-first retry distribution. Lazy grammar activation
inside a proposed block is applied to the subsequent positions. Unknown or
stochastic filters retain the previous verifier. Request scratch is not copied
into sampler checkpoints.

Proposal tuning uses temperature factors 1, 0.85 and 1.15, with bounded feedback,
hysteresis and infrequent decisions. It leaves the target sampler unchanged.
The contextual shortlist defaults to 4096 entries, preserves recent tokens and
refreshes from the accepted target prefix rather than a rejected hypothetical
suffix. Poor initial acceptance restores the complete proposal head. Both paths
record the actual proposal distribution for corrected block verification.
Changing the proposal can change the output for a fixed seed without changing
the intended target distribution.

GPU probability preparation is a separate optional path. The acceptance decision
still runs on the CPU. It requires a compatible stateless chain beginning with
bounded top-k; grammar/reasoning transitions retain full-logit CPU fallbacks.
Its floating-point reductions need not be bit-identical to CPU calculations.
The existing CUDA admission checks require headroom, and allocations remain
subject to the model's managed VRAM budget. The check is not a fixed promise
about total device memory, which also includes the driver and other programs.

## Compact recurrent rollback

The extended path stores two complete F32 states and exact rank-one update
records. Requested intermediate states are reconstructed in the original
arithmetic order. It is sequential rollback, not tree speculation or a new
lower-precision recurrent representation.

Admission currently requires the supported Qwen35/Qwen35MoE Gated DeltaNet
geometry, one sequence, compatible CUDA capability 12.0, F32 recurrent state,
and three or four proposals. The original two-proposal compact path remains
available. Disabling `WILDLLAMA_GDN_COMPACT_ROLLBACK` disables both compact modes.
Canonical checkpoints remain interchangeable with the full-state layout.
The switch does not increase speculation depth automatically.

For the tested 48 recurrent layers with 128-wide state and 48 value heads,
the extended representation saves approximately 283 MiB at three proposals
and 425 MiB at four, compared with full-state rollback at the same depth.
This is state-buffer arithmetic, not a universal percentage of total VRAM.

## Validation and scope

Regression coverage includes deterministic probability previews and continued
RNG state, penalties, active/lazy grammars, CUDA probability error bounds,
Q4_K bitwise kernel comparisons, and compact-state reconstruction, checkpoint
serialization and cross-layout restoration. The CUDA sampler test is explicitly
bound to the built backend, so a missing test backend cannot silently stand in
for a GPU pass.

Full-model performance must be measured with the same context, sampling,
speculation depth and backend. Warm generation is separate from model loading
and prefill. Reject paging-contaminated intervals and distinguish total board
VRAM from the model's own allocations. Short fixtures do not establish answer
quality on all models and tasks.

These changes draw on the ideas of fused GPU work, proposal-only head reduction
and compact recurrent verification. They do not incorporate a trained external
draft model, a packed asynchronous NanoSpec cache, a fully device-resident p/q
verifier or a tree/WY decoder. Experimental fused normalization and static GDN
unrolling that regressed local measurements are excluded from the runtime.
