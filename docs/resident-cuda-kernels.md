# Resident CUDA decode kernels

These guarded decode kernels preserve the model's stored weights, context,
sampling and memory allowances. Validation separates isolated GPU components
from native whole-model decode throughput.

## 095: grouped Q8 attention

The former paired-head launch could select a different split-K partition count
from the legacy per-head launch. That changed the order of floating-point
accumulation, rescaling and final merging, so identical greedy text did not imply
identical probabilities. On the tested NVIDIA path, the value accumulator is
FP32 (`float2`); the HIP-only half accumulator does not explain this result.

The corrected launcher uses the legacy kernel's occupancy, 128-thread reference
block size and logical head count when choosing partitions. The existing split
selector and canonical output/metadata layout remain unchanged. The new kernel
puts two independent 128-thread, four-warp head groups into one 256-thread CTA.
Each group retains its original dot products, accumulation and reductions.
Separate shared-memory banks prevent overlap. Specialization removes generic
sequence, compact-mask, sink and bias branches only where the host gate proves
their values; the original one-group kernel keeps its old expressions.

Automatic activation is restricted to NVIDIA SM120, 24 query heads and four KV
heads. The complete gate still requires dimension 256, one query and sequence,
Q8_0 keys/values, F32 queries, a supported nonnull dense F16 mask, an even grouped
head ratio, valid dimensions/strides, no sinks, zero bias and zero softcap.
Unsupported layouts keep the legacy path.

`WILDLLAMA_CUDA_GQA_Q8` accepts:

- Unset or empty: automatic activation for that measured device/head geometry.
- `0`: legacy path.
- `1`: explicit opt-in on other eligible NVIDIA layouts; all safety guards apply.
- Any other value: legacy path, including whitespace and textual booleans.

CPU policy checks passed, including 7,072 canonical scratch-row mapping checks.
GPU correctness passed all 108 cases bit for bit against the legacy kernel:
the production 24/4 head geometry at 1,280/33,024 KV tokens and the original
4/1, 6/1, 8/2 and 12/2 geometries at 256/512/768/1,024 tokens, each using three
input patterns and contiguous/padded views. The previous automatic launch
reproduced numerical differences in 12 negative controls.

Captured GPU-event measurements bracketed 32 kernel-plus-canonical-merge
iterations, with three warmups and 12 balanced comparison rounds. Allocation,
copies, graph capture/instantiation and host scheduling were outside the timing.
The final expanded run measured component speedups of 1.974x, 1.959x, 2.032x,
2.015x, 2.439x, 3.136x and 3.155x at 256, 512, 1,024, 1,280, 8,192, 33,024 and
65,536 KV tokens respectively, with 12/12 wins at every length. These are
attention-component measurements, not whole-model speedups.

## 096: IQ3_S matrix-vector row pairing

Two independent output rows share a CTA while retaining the original four-warp
K assignment, vector dot implementation and per-row reduction order. Production
eligibility is limited to SM120, IQ3_S weights, contiguous plain 2D matrices,
one F32 input token, F32 output, no gathered expert IDs and no fused operation.
The only enabled weight geometries are input width 5,120 with either 5,120 or
17,408 output rows. Q8 and other quantizations retain their existing kernels.

`GGML_CUDA_IQ3_ROW_PAIR` accepts unset/empty or exact `1` for those eligible
geometries. Exact `0` and malformed values disable it. Explicit opt-in does not
broaden the shape/device gate.

Nine tests using small ranges of the installed model's trained IQ3_S weights and
three activation-scale patterns were bitwise identical. The final isolated
component run measured 1.106x for the 5,120-row square range and 1.043x for the up
projection. The square range was a 5,120-row prefix of a real 10,240-row QKV
tensor, not a full 5,120-row projection from the model. The complete 10,240-row
QKV shape remains on the legacy path. The down projection measured 0.966x and
is explicitly excluded, together with all unmeasured neighboring shapes.
The combined whole-model run below establishes compatibility; it does not
establish an isolated whole-model throughput benefit from `096`.

## Memory and compatibility boundaries

Neither change introduces a global cache, additional weight copies, a new device
allocation or altered quantization, sampling, context or memory limits. Attention
keeps the legacy split count and global scratch sizing. Its two-group CTA uses
8,704 bytes of SM-local shared memory instead of 4,352 bytes and has no measured
local-memory spill. This is shared memory within an SM, not a new persistent
VRAM buffer. IQ3 row pairing uses the existing quantized activation allocation
and lifetime. Existing fusion flags and all other model paths are preserved.

## Whole-model comparison

An isolated Windows/RTX 5060 Ti 16 GiB comparison used the same resident Huihui
Qwen35 27B IQ3_S model, 65,536-token context capacity, Q8 KV, greedy sampling and
128 output tokens. Each role completed eight requests: one warmup and three
warm measured requests at each of 1,024 and 32,768 input tokens. No WildGoose,
model startup or model-loading time is included in these native completion
decode timings.

The baseline used the previous CUDA payload with both additions disabled. The
attention role enabled grouped attention with IQ3 row pairing disabled; the
combined role used automatic selection for both. All roles kept the legacy
dense-fusion opt-in disabled. The reports verified identical model/profile and
prompt identities, eight unchanged common native files and distinct baseline
versus candidate CUDA payload hashes.

| Input tokens | Baseline median tokens/s | Attention | Combined automatic |
| --- | ---: | ---: | ---: |
| 1,024 | 26.946 | 27.478 (+1.98%) | 27.448 (+1.86%) |
| 32,768 | 16.530 | 22.647 (+37.00%) | 22.532 (+36.31%) |

Every accepted case matched generated content, selected token IDs and reported
selected/top-three IDs, ordering and log probabilities exactly. This comparison
covers the recorded probabilities, not every vocabulary logit or all prompts.
The combined and attention medians differ by less than 1%; no additional
whole-model tokens/s improvement is attributed to IQ3 row pairing.

One paging-contaminated attention measurement was discarded and repeated.
All retained timed windows passed the paging checks. This describes accepted
measurements, not an absence of system-wide paging throughout the session.
The performance suite did not run separate functional/tool-call cases. Results
are specific to these prompts, hardware and payloads, rather than a universal
speedup guarantee.
