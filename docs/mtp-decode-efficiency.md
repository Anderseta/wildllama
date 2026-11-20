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

## Adaptive four-token MTP and code editing

With a single compatible MTP slot, set `draft_num_predict` to 4 and
`speculative_adaptive` to true to compare two and four proposals within each
request. The controller starts at two and measures elapsed time per committed
token, including rejected drafts and rollback. It briefly probes the other
depth throughout generation and switches when the measured cost is at least
5% lower. Evidence is kept separately for different context lengths. The
server checks rollback support and the available rollback depth before using
this policy. With `WILDLLAMA_CUDA_SYNC_TRACE=1`, the end-of-request log
includes the number of measured rounds at each depth.

For single-slot Qwen35 MTP on SM120 with Q8_0 KV and the adaptive four-token
profile, the launcher also enables these native runner options:

| Runner environment option | Effect |
| --- | --- |
| `WILDLLAMA_MTP_EDIT_LOOKUP` | Matches six confirmed tokens against the last 32,768 text prompt tokens and proposes a matching continuation. The target verifies each proposed token with its actual deterministic proposal probability. Cost is up to 144 KiB of host index and token storage per eligible slot. |
| `WILDLLAMA_MTP_REUSE_TARGET_FENCE` | Reuses an immediately preceding target synchronization when the MTP consumer reads already-ready host embeddings. Other backends and state layouts retain their required waits. |

Both accept an explicit runner override of `0`. The target weights, logits
and sampling settings stay in use. With stochastic sampling, different
proposal sequences can consume random draws in a different order. Peak VRAM
also depends on the configured maximum rollback depth.

During a code edit, the lookup keeps its position in the source while the
target continues to confirm the same tokens. Before reusing the position,
it checks the anchor and every newly committed token, including a possible
bonus token. A mismatch or a rewind sends the lookup back to the index.
This keeps repeated function bodies and route definitions aligned with the
occurrence being edited. The cursor uses three machine words per lookup.

For vocabularies of at least 4,096 tokens, CPU sampling and MTP block verification
can select candidates directly from the raw scores when the first effective
filter is top-k with k between 1 and 127. The bounded heap retains one extra
candidate to detect ties at the cutoff. On SSE2 CPUs, blocks of 16 finite
scores below or equal to that cutoff can be skipped together. The regular
sampling chain receives the surviving IDs and original scores.

Ties in the selection, nonfinite scores, active grammar constraints, forced
reasoning tokens and requests for token probabilities take the full-row path.
After three consecutive ambiguous rows the sampler waits 16 eligible rows
before trying again. The selection uses host memory only.

For edit drafts, a request-local index retains the two latest positions per
anchor bucket. It verifies the six anchor tokens and compares up to 32 earlier
tokens for ambiguous matches. The target still verifies the resulting proposal.

CUDA runners can opt into exact single-row top-k readback using
`WILDLLAMA_EXACT_GPU_TOP_K=1` in the runner environment. With a supported leading
top-k from 1 through 127 the CPU samples original scores gathered from the GPU.
Ties, unsupported sampler chains, probability requests and uncertain scores use
the full-row path. Set `WILDLLAMA_TRACE_READBACK=1` to log cumulative raw score
bytes transferred eagerly, deferred and subsequently materialized. Benchmark
with the intended model and sampling settings before enabling the option.

## Experimental per-request MTP controls

Single-sequence MTP can opt into three independent runner environment settings.
All default to 0. The target sampler still verifies proposals with their actual
distribution. Changing a proposal may change a seeded output.

| Runner environment setting | Effect of setting it to 1 |
| --- | --- |
| `WILDLLAMA_MTP_RESPONSE_LOOKUP` | Indexes six-token anchors in the last 16,384 confirmed response tokens on demand. The index uses 16 KiB of host storage and reads the existing token history. Prompt matches retain priority. Slot rewinds and new requests reset the index. |
| `WILDLLAMA_MTP_AUTO_STRATEGY` | For eligible text requests with stochastic block verification, compares the ordinary MTP proposal, proposal-temperature tuning and a contextual 4,096-token shortlist. Uses elapsed time per committed token across consecutive rounds. A candidate must beat the incumbent by 6% in the moving estimate. The shortlist uses the existing CUDA Q4_K-head admission checks. A fixed shortlist or a forced temperature, dynamic vocabulary or packed-head mode takes precedence. |
| `WILDLLAMA_MTP_EARLY_STOP` | May verify after two proposals when confidence is low and recent uncut blocks rarely yielded a third accepted token. Every fourth eligible decision probes the full depth again. Uses probabilities already available on the host and respects the configured maximum. |

The last two switches need independent and combined measurements on the intended
model. The response lookup does not require a GPU index or new weights.

When automatic proposal selection and short adaptive MTP depth are both in
use, depth trials use ordinary proposals. Strategy trials hold a learned depth
and have separate cost histories by depth and context range. Experimental
strategy timing does not enter depth estimates. Both use elapsed time per
committed token and the model's existing verification and sampling paths.
