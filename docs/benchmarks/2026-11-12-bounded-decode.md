# Bounded decode experiments — 2026-11-12

## Decision

Ship three independent opt-in profile controls. Keep all three off by default.
Keep the existing Huihui MTP profile, context, sampling and memory limits.
There are useful kernel-level wins, but no consistent whole-model win across
both workloads sufficient to replace the current defaults.

## Setup and measurements

Windows, Ryzen 5 2600, RTX 5060 Ti 16 GiB, CUDA 12.8. The existing Huihui Qwen
27B mixed IQ3_S GGUF was used without conversion. Context stayed at 40,960,
initial physical KV at 8,192, Q8 K/V, one sequence, temperature 0.6, seed 149,
top-k 20, top-p 0.95 and 256 output tokens. MTP comparisons used two proposals.
No model was downloaded or copied. Model loading was excluded from decode timing.

Eleven isolated configurations produced 82 retained generations. Every timed
window passed the paging monitor; contaminated attempts were retried. Physical
RAM and Windows commit headroom were monitored independently. The lowest observed
available RAM remained above 11.7 GiB; commit headroom stayed above 8.6 GiB.
The production profile and manifests were not changed by the measurements.

The confirmation round used one warm-up and four timed generations per prompt.
Values below are median tokens/s and the full timed range. N-gram rows are the
earlier two-timed-repeat comparison, against the same mode with feedback off.
They use the existing n-gram proposal budget; the feedback table additionally
caps its own proposals at eight. N-gram modes do not load the MTP head.

| Configuration | Portuguese prose: tokens/s (range) | Python: tokens/s (range) | Peak whole GPU, MiB |
| --- | ---: | ---: | ---: |
| Installed runtime, MTP 2 | 43.29 (37.40–43.91) | 43.21 (42.17–43.32) | 13361.4 |
| Candidate, new options off | 40.40 (35.26–43.60) | 43.34 (41.95–43.62) | 13360.5 |
| Token reuse | 43.05 (42.57–43.44) | 42.65 (41.22–43.43) | 13376.4 |
| L2 activation policy | 44.19 (43.51–44.68) | 43.27 (42.07–44.01) | 13354.9 |
| Existing ngram-simple | 29.26 (29.19–29.34) | 29.59 (29.27–29.90) | 12723.0 |
| Feedback proposer | 28.30 (28.27–28.34) | 27.52 (26.81–28.23) | 12711.5 |

Startup/warm-up drift was material, including in the installed runtime. Earlier
baseline runs ranged from roughly 33 to 40 tokens/s while later warmed samples
reached 43. Do not present their difference as an optimization gain. Token reuse
and L2 did not improve both workloads consistently beyond that variation. The
feedback proposer was slower than existing n-gram decoding on these prompts.
All MTP comparisons, including the installed runtime, produced identical text
and accepted-token counts for the same prompts and seed.

## Kernel and functional checks

- 48 dense/routed cases compare candidate and reference output bits, including
  output guards, odd row/token counts, matching/different routes and input strides.
  The largest output-head case is the real 5,120 × 248,320 shape.
- 83 filtered backend tests compare CUDA with CPU while both CUDA switches are on.
- The CPU proposer test covers rejection feedback, relearning, sequence isolation,
  reset, zero/short budgets, oversized feedback and hash-collision churn.
- Go checks validate all three boolean runner environment keys; UI tests cover
  independent toggles, persistence and capability fallback. TypeScript also passes.

In CUDA-event paired tests, small dense Q4_K batches reduced kernel time by
roughly 8–32%. Larger routed projections with matching experts saved roughly
12–15% in several cases. Different experts could be 22–32% slower; the large dense
case was about 10% slower and the full output-head case showed no useful gain.
These are synthetic operation timings, **not** whole-model tokens/s gains.
The final dispatch retains the original kernel for single-token MoE decode,
which has no cross-token reuse opportunity. Its regression tests are rerun on
the final package; that guard does not change the measured dense Huihui path.

The new paths do not copy weights into another VRAM cache. The proposer owns
74,064 bytes of CPU state per sequence. L2 uses at most 1 MiB of existing on-chip
cache, not newly allocated VRAM. Paired MTP peaks remained around 13.05 GiB of
whole-GPU usage; small differences of several MiB are not evidence of a new
model-sized allocation. Verification buffers still depend on draft shape and
the existing runtime budgets.

No large SSD-offloaded MoE throughput claim is made from these component tests.
Its real routing overlap and storage bottleneck require a separate model run.

See [runtime controls](../bounded-decode-experiments.md) and the
[compact measurement summary](2026-11-12-bounded-decode.json). Raw logs, commands,
payload hashes and the deployment inventory remain in the local diagnostics
directory `G:/Goose/diagnostics/20261112-modchip-adaptations`.
