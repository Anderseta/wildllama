# MTP verification and proposal efficiency

WildLlama keeps the target model's weights, tokenizer, sampling settings and
context unchanged. These changes reduce work in short speculative batches and
in proposal selection. They do not change the model's quantization.

## Enabled changes

Compatibility patch `098` extends the validated IQ3_S row-tiling kernel to
two- and three-token batches. It uses four independent output rows per CUDA
block while preserving the original K assignment, dot product and four-warp
reduction. The single-token path retains its existing two-row layout.

Dispatch is limited to SM 12.0, contiguous plain IQ3_S matrices, and the measured
matrix/batch combinations. Shapes that were slower retain the previous kernel.
Other quantizations, devices, fused operations and expert-indexed operations
retain their existing paths. No persistent GPU buffer or duplicate weights are
introduced. The larger tiles use more registers and block-local shared memory;
the tested kernels have no register spills.

`GGML_CUDA_IQ3_ROW_PAIR_BATCH=0` disables only the new short-batch dispatch for
diagnostics. `GGML_CUDA_IQ3_ROW_PAIR=0` also disables the prior single-token
dispatch. `GGML_CUDA_IQ3_ROW_PAIR_TRACE=1` reports actual dispatch once per shape.

Patch `099` selects only the highest-scoring proposal when MTP uses greedy
sample-and-match without a confidence cutoff. The former ten-candidate
selection was unnecessary in that case. Probabilistic rejection and positive
confidence cutoffs retain their original top-ten proposal distribution. This
does not change the user's target-model top-k or temperature.

## Indexed proposal head: opt-in experiment

`WILDLLAMA_MTP_DRAFT_VOCAB=<count>` enables an experimental proposal-only indexed
head. It is **off by default**: reduced proposal acceptance outweighed the saved
head work in the measured Huihui workload. There is no recommendation to enable
it for that profile.

The experiment supports one sequence, Qwen3.5/Qwen3.5-MoE single-head MTP, and a
plain, unscaled Q4_K output matrix on one CUDA device. Adapters, Hadamard
transforms, tensor splitting, other head formats and unsupported operations use
the full-head fallback. Counts must be between 256 and 131072, below the model's
vocabulary size; the host index also has a fixed vocabulary bound.

Selected rows are views of the existing quantized head, not copied or
dequantized weights. Global token IDs are retained. The target always evaluates
the complete vocabulary and verifies proposals. Selection changes can affect
acceptance, timing and fixed-seed trajectories; they do not authorize the draft
model to bypass target verification.

When full target logits are already available on the host, they seed the
selection; otherwise a BPE prefix and recently observed tokens are used.
`WILDLLAMA_MTP_DRAFT_VOCAB_SEED=0` selects the latter policy explicitly. The
implementation never downloads the full target output just to seed this list.
It refreshes graph inputs on every replay and falls back to the complete head
after very low early acceptance. CUDA operations are chunked below the grid-Y
limit, including the 65536/131072-row cases. Additional index and scratch tensors
still consume some memory and remain subject to the existing allocator limits.

## Validation, 2026-11-10

Hardware: RTX 5060 Ti 16 GiB, CUDA 12.8, Windows. Model:
`huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`, IQ3_S body with a shared Q4_K
output head. Native base: `9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e`.

The production-kernel fixture checked 72 trained-weight cases: eight shapes,
one/two/three tokens, and three activation patterns. Results were bitwise equal,
including an odd-row tail outside production dispatch. CUDA-event measurements
showed approximately 3–14% gains for enabled short-batch combinations. The
fixture launches the previous kernel with its actual rows-per-block geometry;
it does not compare against an artificially oversized grid.

The indexed-head fixture uses the same graph helper as production. Eight
replays at both 512 and 131072 selected rows gave bitwise-equal selected logits
against the complete head; excluded logits remained negative infinity. Weight
fixtures were below 32 MiB. CPU MTP-state, sampler-output and shortlist tests
passed, including grammar, probabilities, RNG, checkpoint and fallback checks.

An isolated full-model comparison against `8c45bece` used the existing profile:
40960 context tokens, Q8_0 KV, MTP with two proposals, temperature 0.6 and seed
149. Each Portuguese prompt generated 256 tokens three times; the first run was
excluded from the throughput average. Indexed heads were disabled.

| Prompt | Installed tokens/s | New tokens/s | Difference |
| --- | ---: | ---: | ---: |
| Library organization | 43.20 | 43.28 | +0.18% |
| Python transactions and tests | 38.25 | 38.76 | +1.34% |

All six responses, reasoning text and proposal/acceptance counts matched.
No paging was detected in those runs. Physical RAM remained above 11.1 GiB free.
The total-device peaks were 13.77 and 13.66 GiB, respectively, but background GPU
use differed: these figures are not evidence of a VRAM saving. Enabled changes
add no persistent GPU allocations. The end-to-end gain is modest and the prosaic
case is within ordinary timing noise; isolated kernel gains must not be reported
as whole-model speedups.

The 32768-row head experiment reduced acceptance and was slower. At 131072 rows,
there was no convincing overall gain. Both remain disabled in the installed
profile. Earlier interrupted runs near the Windows commit limit were discarded;
reaching that limit is not itself proof of disk swapping.

## Short verification and recurrent-state follow-up

The Q8 verification attention path in patch `100` handles two- and
three-query batches on the measured Blackwell geometry. Two queries share
key loads while keeping independent accumulations and the previous split-K
layout. Three queries keep the previous MMA attention arithmetic and combine
the Q8 K/V conversions into one launch using the existing scratch buffers.
The grouped and conversion paths can be tested with
`WILDLLAMA_CUDA_GQA_Q8_VERIFY=1`.

Patch `101` adds a full-vocabulary Q4_K projection for two- and three-token
verification on SM120 for the measured 5,120 by 248,320 matrix. A warp computes
one output row and preserves the original four partial sums and final
reduction. The kernel uses the original quantized weights and activation
buffer. The production switch is `GGML_CUDA_Q4_HEAD_WARP=1`. Single-token
projection retains its previous dispatch.

Patch `102` lets an eligible Gated DeltaNet kernel read the indexed recurrent
state row directly and use the existing fused cache write. The graph requires
a private gather with the expected dimensions, the same recurrent allocation
for reading and writing, and no conflicting aliases. Unsupported shapes,
placements and dependencies keep the ordinary gather. Use
`WILDLLAMA_CUDA_GDN_GATHER=1` to test this path.

The separate compact-state fixture reconstructs the middle rollback state
from a full state and an update record in the same FP32 operation order.
All 36 state and alias cases matched bit for bit. For this 48-layer geometry,
the two full states and record would save 141.74 MiB against three full states.
This is a size calculation from the fixture, with no change to runtime cache
allocation. Checkpoint serialization, prefill, context shifts, and restore
would need to handle the compact representation before runtime activation.

The existing `speculative_adaptive` option can use a request-local controller
for single-sequence MTP with a ceiling of one, two, or three proposals.
Patch `103` measures time per committed token across depths from zero through
that ceiling, including rejection cleanup. It keeps separate observations for
different context lengths and periodically retries an alternative depth.
It uses the previous controller where mixed speculation, unsupported rollback
capacity, synthetic acceptance, a minimum draft length above one, or response
overlap makes the comparison unsuitable. The policy adds no GPU storage.

The WildLlama launcher selects the three verified CUDA paths for one SM120
device, one sequence, Qwen35, Q8_0 KV and an MTP ceiling of one or two proposals.
Explicit environment values take precedence. The native tensor and graph checks
still control dispatch. The adaptive policy follows the existing profile option.

### Follow-up measurements

The two-query Q8 attention fixture was 1.78–2.42 times as fast as the prior
kernel. Three-query conversion was 1.10–1.36 times as fast on short contiguous
inputs and 1.53–2.47 times as fast on padded inputs. Long contiguous conversion
times were equivalent. All 128 compared attention and conversion cases matched
bit for bit, including the 108 existing cases.

The Q4_K fixture read 90 MiB of trained weights. Two- and three-token projections
were 1.44 and 1.47 times as fast in the final paired trial. All 12 comparisons
matched bit for bit, including ties and odd row counts. The GDN graph fixture
matched states and scores in 27 replays and three fallback controls. Direct
gather reduced the one- and two-token component times by about 20% and 16%.
Three-token times were equivalent.

The full Huihui comparison kept the 40,960-token capacity, Q8 KV, two proposals,
temperature 0.6, seed 149 and 256 generated tokens per request. The first run
was excluded from each generation-speed average.

| Prompt | Installed tokens/s | Candidate tokens/s | Change |
| --- | ---: | ---: | ---: |
| Short Portuguese prose | 43.71 | 44.08 | +0.86% |
| Short Python task | 39.06 | 39.49 | +1.10% |
| Repetitive 16,384-token input | 53.67 | 54.85 | +2.20% |

The short candidate measured attention and Q4_K. A localized MSVC include
message caused Ninja to retain two stale Unity objects in that build. Rebuilding
both objects resolved the issue. The long-input candidate traces confirmed all
three CUDA paths, including GDN for one, two and three tokens. The nine paired
fixed-depth responses and proposal counts matched. The comparisons detected no
paging. The logged CUDA KV, recurrent-state and compute reservations matched
between long-input runs. Whole-device peaks were 14,225 and 14,242 MiB, including
desktop activity.

The first short candidate run had slower prompt evaluation, 8.07 seconds
versus 0.96 seconds. The later long-input cold prefill measured 19.12 seconds
versus 19.17 seconds. The table measures warm generation. One adaptive request
per prompt produced 39.94, 39.85 and 52.39 tokens/s respectively, so the default
profile retains fixed depth. The tiny HTTP fixture exercised adaptive depth
selection and matched all 96 reference token IDs.
