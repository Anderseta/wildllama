# Bounded inference efficiency

Patches 067-071 continue the SGLang-inspired work inside the existing native
engine. They preserve original weights, quantization, logical context and saved
profiles. They do not establish performance parity with SGLang.

| Component | Change | Activation |
| --- | --- | --- |
| Prompt scheduling | Learn marginal prompt cost and admit bounded slices beside generation, with fairness and the existing memory-growth guard. | Existing `WILDLLAMA_ADAPTIVE_PREFILL=1` experiment. |
| Speculative rejection | Direct dense token lookup replaces sorting/searching; residual entries omit an unused logit. | Automatic for eligible supports; exact fallback otherwise. |
| Cold context tier | Completed prewrites release their worker tickets; warm entries retain their backing and avoid rewriting the same snapshot. | Existing explicitly configured disk/async tier. |
| KV branch sharing | Copy sequence memberships through the existing position index instead of scanning the complete logical cache. | Automatic for same-stream sequence copies. |
| Quantized CUDA projections | Share the same immutable Q8_1 activation across compatible Q4, Q5, Q8 and K-weight projections, including mixed weight types. | Existing `GGML_CUDA_DENSE_OPT=1` experiment. |

See [prompt slicing](adaptive-prefill-chunks.md),
[sampling](sampling-pipeline-efficiency.md) and
[warm/cold handoff](state-cache-efficiency.md) for their complete contracts.

## Indexed KV branches

Same-stream sequence copies now visit only source cells in the requested
position interval. An existing destination membership is left intact, so repeated
copies remain idempotent. Cell positions, spatial coordinates, shifts, physical
occupancy, other sequences and the next-placement hint are preserved. Source
index iterators remain valid while a different destination index is updated.
Cross-stream tensor copies retain the existing implementation.

This removes a logical-capacity scan from branching and shared-prefix work. It
does not add a KV tensor or reserve more cells. The existing cell allocator and
attention kernels are unchanged; this is not a universal paged-KV/radix system.

## Shared quantized activations

The ordinary CUDA MMVQ input quantizer already emits the same `block_q8_1`
representation independently of the selected standard Q/K weight type. Compatible
consumers can therefore reuse its exact bytes while keeping their own original
matrix kernels and accumulation order. No weights are converted or requantized.

The existing planner still proves input identity and immutability, rejects
intervening aliases/custom writes and excludes concurrent streams and routed-ID
matrices. Existing stream ordering, graph capture and reverse allocator release
remain in force. Reuse ends after the last consumer. The logical retained-input
bound stays 4 MiB; allocator granularity and every tracked CUDA allocation remain
subject to the managed device quota. This is not a total-device memory bound.
Unsupported formats and layouts keep their previous paths. The experimental
switch and saved profile values are unchanged.

## Validation and remaining architecture work

The validation record distinguishes exact CPU contracts, tiny CUDA operator
comparisons and component timings. GPU measurements made while other programs
use the device do not establish a trained-model throughput improvement.

The [validation record](benchmarks/bounded-inference-efficiency-20261102.json)
contains 11 passing CPU targets, 12 before/after HTTP responses and four
concurrent HTTP requests with actual adaptive scheduling. The latter kept the
same greedy SSE fields and admitted eight bounded prompt slices without primary
or secondary compute growth. The synthetic fixture is approximately 28 MiB;
its declared context was overridden equally in both controls, without modifying
the file or any user profile.

The tiny CUDA projection comparison covers ten quantizations with one/four
tokens plus mixed Q/K weights. Enabled and disabled reuse produced 207,360
bit-identical output bytes. The observed managed peak was 2,194,944 bytes in
both processes under a 64 MiB test quota. Traced submissions/captures recorded
42 shared quantizations and 126 reuses. These are correctness and component
observations; tracing, cold initialization and other GPU users preclude a
throughput claim from the process wall times.

General shared KV paging across memory classes, page-level hierarchical context
movement, a broader hybrid/speculative scheduler and a fully device-resident
verification/commit pipeline remain larger architecture work. Existing aliases,
serialized snapshots, graph arenas and response overlap should not be described
as those complete systems. The rejected direct-token relay remains excluded.
