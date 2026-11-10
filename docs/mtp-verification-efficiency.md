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
