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
Patch `096` enabled input width 5,120 with either 5,120 or 17,408 output rows.
Patch `097` extends this gate to the four measured projections below.
Q8 and other quantizations retain their existing kernels.

`GGML_CUDA_IQ3_ROW_PAIR` accepts unset/empty or exact `1` for those eligible
geometries. Exact `0` and malformed values disable it. Explicit opt-in does not
broaden the shape/device gate.

Nine tests using small ranges of the installed model's trained IQ3_S weights and
three activation-scale patterns were bitwise identical. The final isolated
component run measured 1.106x for the 5,120-row square range and 1.043x for the up
projection. The square range was a 5,120-row prefix of a real 10,240-row QKV
tensor. The complete 10,240-row QKV shape was subsequently measured in `097`.
The down projection measured 0.966x and
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

## Additional IQ3_S projections and streaming

Patch `097` extends single-token, two-row dispatch on SM120. Each new shape
won all 12 paired GPU-event comparisons on an RTX 5060 Ti. The fixture reads
the indicated trained matrix, quantizes three input patterns and compares
every output bit with the original kernel. The table measures kernel execution.

| Input width | Output rows | Projection in the Huihui model | Kernel speedup |
| ---: | ---: | --- | ---: |
| 5,120 | 10,240 | Combined QKV | 1.111× |
| 5,120 | 12,288 | Query | 1.113× |
| 6,144 | 5,120 | Attention output | 1.101× |
| 5,120 | 1,024 | Key or value | 1.071× |

The fixture also exercises the previous shapes and the excluded down
projection. All 21 comparisons matched bit for bit. These additions reuse the
existing activation buffer. Multi-token, fused and other device paths keep
their original dispatch.

On the host, plain completion streams emit token chunks directly. Structured
chat streams keep the incremental parser and compare its new message against
the current message before moving it into place. Streaming avoids building a
second vector of token IDs and a probability history for the final event.
Requests for pre-sampling log probabilities keep the same partial sort, FP32
sum order and selected-token probability, including tokens outside the top
list. They normalize only entries returned to the client.

The Windows comparison against `83745f24` used the same Huihui GGUF, RTX 5060
Ti, 65,536-token capacity, Q8 KV and temperature 0.6 with a fixed seed.
Each median covers three warm streams of 160 generated tokens. The 32,768-token
workload started a fresh server for each version to isolate it from earlier
chat and tool requests.

| Input tokens | Previous tokens/s | Updated tokens/s | Previous / updated peak device use |
| ---: | ---: | ---: | ---: |
| 1,024 | 29.272 | 29.292 | 13,956 / 13,936 MiB |
| 32,768 | 24.128 | 24.106 | 14,730 / 14,730 MiB |

The decode medians differ by less than 0.1%. Device use includes the desktop.
Both versions reported 1,152 MiB of device KV at the long input. All 14 paired
requests matched on their recorded output fields, including streamed token IDs,
selected/top-three log probabilities, Portuguese chat deltas and a tool's name
and arguments. One candidate timing window was repeated after a paging signal.
All retained timed windows passed the same paging check.

A separate CPU replay fed 8,192 six-byte chunks through the server's update
and JSON serialization routines. Median time over five measured replays fell
from 5,395 ms to 39 ms for plain completions. Structured chat measured 5,345
and 5,402 ms respectively. This isolates server-side work on a long response
and excludes model generation and network transport.
