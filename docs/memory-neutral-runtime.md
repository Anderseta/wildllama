# Memory-neutral runtime follow-ups

Patches 055 and 056 remove callback type erasure and replace a small sampler graph
fragment. They preserve model weights, quantization, context, sampler ordering,
and configured memory limits. They apply to dense and MoE models through the
shared runtime; they do not change expert selection or weight offloading.

This document describes the implementation and its validation gates. No new
model throughput result or performance parity with SGLang is claimed here.

## Scope and defaults

| Patch | Change | Activation | Memory contract |
| --- | --- | --- | --- |
| 055 | Fused inclusive top-p mask after the existing CDF | Automatic for standard CPU support up to 1024 and NVIDIA CUDA support up to 256 candidates; explicit comparison control available | One ordinary F32 result, eligible reuse of a dead input, and bounded on-chip reduction storage. No new persistent GPU workspace or device scratch allocation. |
| 056 | Concrete server-slot visitors replace callback type erasure | Automatic | Existing callback ownership and slot storage; no new GPU buffers or retained runtime cache. |

These changes do not enable optional graph arenas, expand KV capacity, change
cache budgets, or reserve additional model memory. Those settings retain their
existing independent defaults and limits.

## Bounded top-p mask fusion

The backend still performs the same argsort, candidate gathers, softmax, and
cumulative sum. Fusion starts only after that CDF exists. It replaces the
scale/step/sum/clamp/index-update/log/add mask fragment with one operation,
implemented as mode 2 of the existing `GGML_OP_TOKEN_MASK`.

For each CDF entry, the mask uses the original floating-point expression:

```text
keep[i] = (-1 * cdf[i] + p) > 0
inclusive = clamp(sum(keep), 0, count - 1)
keep[inclusive] = true
result[i] = logit[i] + (keep[i] ? +0 : -infinity)
```

The inclusive position is the number of positive mask entries, not an assumed
first threshold crossing. That distinction preserves the old graph even for
nonmonotonic or nonfinite CDF input. The final addition is also retained:
replacing a rejected logit directly with negative infinity would change cases
such as positive infinity plus negative infinity, or the sign of zero.

Admission requires contiguous F32 vectors containing 1-1024 candidates, finite
`p` in `[0, 1]`, and `min_keep` at most one. The backend must explicitly support
the new operation mode. Initialization checks that support with tensor metadata
only; it allocates no GPU buffer. Larger support sets, other parameter ranges,
and unsupported backends retain the literal legacy graph. Existing CPU sampler
behavior outside the backend graph is unchanged.

The CUDA implementation counts in registers or a bounded shared integer array and then
writes the existing result. The CPU implementation completes both passes on one
worker. Neither implementation allocates a device workspace. The allocator may
reuse a dead logits or CDF buffer only through its existing shape, ownership,
reference-count, view, and output-protection checks. This permission is specific
to mode 2; grammar-mask modes 0 and 1 retain their existing allocation behavior.

The sampler's admitted candidate limit selects the optimization only; it does
not truncate the candidate set. For example, top-k 512 still retains 512
candidates and uses the legacy CUDA mask under the automatic policy. Logical
context, output-token limits, and sampler settings remain unchanged.

The sampler's admitted candidate limit is immutable backend configuration. It contains no
graph tensor pointers or logical RNG/history state. Clones initialize their own
backend capability; graph bindings and state copies continue to preserve their
existing ownership rules.

### Diagnostic control

Set this environment variable before the model runner initializes its samplers:

| Value | Behavior |
| --- | --- |
| Unset | Automatic fusion for the standard CPU buffer up to 1024 candidates, and a backend registered as `CUDA` up to 256 candidates. Other backends retain legacy. |
| `0` | Explicit legacy control on every backend. |
| `1` | Try fusion up to 1024 candidates on a backend that supports mode 2, including the larger CUDA comparison range. Unsupported shapes or parameters still use legacy. |
| Any other value, including an actually present empty string | Legacy graph. |

On Windows, `_putenv_s(name, "")` and common commands for clearing a variable
remove it rather than retain an empty value. Removal selects the automatic
policy. Use the explicit value `0` to disable fusion reliably. The test verifies
this CRT distinction; it does not claim to retain an empty Windows variable.
On POSIX, an environment variable set to an empty string remains present and
selects legacy.

`WILDLLAMA_TOP_P_MASK_FUSION` is an internal comparison control, not a new
per-model API option or a replacement for configured memory limits. Changing it
does not rebuild already initialized sampler graphs.

The automatic ranges follow repeated small-fixture measurements: CPU full-graph
comparisons improved across the admitted range; CUDA comparisons improved
consistently up to 256 candidates. The larger CUDA range remains explicit
because the 512-candidate full graph showed a repeatable regression despite a
faster isolated mask. A larger size improving in isolation does not justify
enabling the entire larger range automatically. These component observations
do not establish a model tokens-per-second improvement.

## Server-slot visitor

The server uses the same slot vectors, traversal order, callbacks, and error
handler. Callbacks remain owned by value. A `std::exception` still reports the
same per-slot error, releases that slot, and permits traversal to continue.
Other exceptions retain their existing propagation behavior.

The change removes the intermediate `std::function` wrapper at the existing
call sites. A large captured callback can otherwise require a temporary heap
allocation for type erasure. A callback already fitting the wrapper's inline
storage might have no such allocation to remove. This affects CPU dispatch and
callback ownership costs; it does not change batching, admission, GPU scheduling,
model kernels, or inference precision.

## Validation and default admission

The dedicated tests compare against frozen pre-change algorithms and literal
legacy GGML graphs. They are designed to check:

- Candidate order, ties, support size, probabilities, seeded draws, reset, and
  observable sampler outputs.
- Exact boundaries, nonmonotonic CDFs, infinities, NaNs, subnormals, and signed
  zero. Finite results, infinities, and signed zero compare bitwise; NaNs compare
  by classification because payloads are not portable across backends.
- Actual in-place reuse of each input, protected graph outputs, and inputs with
  later consumers.
- Backend capability rejection and legacy fallback outside the admission range.
- Identical constrained and unconstrained complete sampling chains, including
  graph bindings and RNG continuity.
- Fewer graph nodes and **actual graph-allocator compute-buffer bytes no greater
  than the legacy graph**, on both CPU and CUDA fixtures. Absence of an explicit
  `cudaMalloc` alone is insufficient evidence for this gate.

`test-top-p-mask-fusion` runs CPU fixtures; `--cuda` requests the CUDA fixtures.
Its optional `--bench` compares warmed legacy and fused component/full-graph
execution across eight ABBA cycles, with 16 raw samples per control and median
timings. Benchmark inputs remain protected between
iterations. `test-server-slot-iteration` checks traversal, ownership, exception,
and allocation contracts; its optional `--bench` measures that component.

Component timings and byte counts are recorded before changing the automatic range.
They do not predict model tokens per second. Small synthetic fixtures establish
correctness and memory bounds without loading large models; real workloads
still determine the practical throughput benefit.

## Measured components

The final Windows / RTX 5060 Ti run used model-free sampler graphs and a
synthetic slot visitor. Each sampler median has 16 samples per control across
eight ABBA cycles. These are component times, not trained-model token rates.

| Component | Legacy | Optimized | Managed compute bytes, legacy / optimized |
| --- | --- | --- | --- |
| CPU sorted top-p graph, 20 candidates | 17.025 us | 9.016 us | 384 / 384 |
| CUDA sorted top-p graph, 20 candidates | 32.431 us | 24.946 us | 768 / 512 |
| CUDA sorted top-p graph, 40 candidates | 38.912 us | 28.398 us | 1152 / 1024 |
| CPU visitor, 20,000 traversals of eight slots | 1473.200 us | 337.150 us | No GPU allocation |

The top-p mask fragment falls from eleven graph nodes to one; the measured
sorted/softmax/CDF/mask graph falls from seventeen to seven. All tested compute
buffers and complete sampling-chain allocations were no larger than their
legacy controls. The visitor's separate large-capture allocation check records
4096 temporary allocations versus zero. Neither change creates persistent GPU
storage. These counters do not measure all driver or process VRAM.

The [validation record](benchmarks/vram-neutral-20261031.json) records measured
medians, activation boundaries, memory comparisons and rejected cases. Final
validation passed 40 CPU and 13 CUDA CTests, 23 native helper/family steps,
300 default API cases per payload, 127 DRY requests on each device, and the
CPU/CUDA streaming cancellation and recovery checks. Only local synthetic
fixtures up to 64 MiB were used; no large or trained model was loaded.

## Rejected CPU scratch experiment

A separate CPU partial-sort experiment packed bucket IDs into bytes and moved
small bookkeeping arrays out of heap allocations. Correctness checks passed,
but repeated timing results varied. The final largest-vocabulary benchmark was
8.7% slower than its unchanged reference, so the experiment was withdrawn.
The shipped CPU bucket helper retains its original storage and sorting logic.
Patch number 054 remains reserved and is absent from the applied stack.

This result concerns that measured component and workload; it is not an
end-to-end model throughput figure. A smaller temporary allocation alone was
insufficient reason to ship a change with a reproducible performance concern.

## Deferred work that can consume more VRAM

Additional retained CUDA graph arenas, larger cached workspaces, more KV
capacity, and speculative/draft output buffers remain separate work. They must
have explicit budgets and independent correctness and performance evidence.
These patches do not enable or enlarge those features. A GPU-side compact draft
result was also deferred because its extra output storage needs a separate
memory and output-contract design.
