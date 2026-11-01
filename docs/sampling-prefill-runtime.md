# Sampling and prefill runtime controls

Patches 057-059 target CPU preparation, a small sampling graph fragment, and
latency when prompt processing competes with ongoing generation. They preserve
the model, quantization, logical context, configured batch limits, and saved
per-model settings. They do not approximate expert routing or prune model
weights.

Draw-tail fusion is automatic for the standard CPU backend within its admitted
range. CUDA draw-tail fusion, MoE preparation reuse, and adaptive prefill require
an explicit opt-in. Component improvements do not establish model throughput
gains or performance parity with another inference engine. Default admission
requires the correctness, memory, and repeated comparisons described below.

## Controls and initial defaults

Set these controls in the native runner's environment before initialization.
They are internal comparison controls, not new request parameters. Changing a
sampler control does not rebuild an already initialized sampler.

| Control | Unset | `0` or any present value other than exactly `1` | Exactly `1` |
| --- | --- | --- | --- |
| `WILDLLAMA_DIST_TAIL_FUSION` | Automatic on the standard CPU backend for supported contiguous F32 vectors with 1-1024 entries; CUDA and other backends retain the original tail | Original categorical-draw tail | Try the fused tail for 1-1024 entries on a backend that explicitly supports it |
| `WILDLLAMA_MOE_PREPARE_REUSE` | Original per-job CPU graph preparation | Original per-job CPU graph preparation | Bounded CPU preparation reuse inside the existing CUDA dynamic MoE prefill provider |
| `WILDLLAMA_ADAPTIVE_PREFILL` | Original server prompt admission | Original server prompt admission | Conservative latency-based whole-prompt admission for eligible concurrent workloads |

For draw-tail fusion, an actually present empty value selects the original
tail. Windows `_putenv_s(name, "")` and common environment-clearing commands
remove the variable instead; removal restores automatic CPU admission. Use
the explicit value `0` to disable reliably. On POSIX, a present empty value
remains distinct from an unset variable and retains the original tail.

`WILDLLAMA_MOE_PREPARE_TRACE=1` additionally reports preparation counters after
completed provider work. Trace output is available with reuse either enabled
or disabled; leave it disabled during performance comparisons.

`WILDLLAMA_PREFILL_TRACE=1` enables correctness-only memory observation with
adaptive admission either disabled or enabled. It records actual primary
compute, primary GPU compute, and secondary retained compute peaks at completed
boundaries, then emits one idle memory summary. Leave this trace unset during
quiet performance comparisons.

`WILDLLAMA_PREFILL_LATENCY_US` optionally supplies an integer latency objective
from 100 through 60000000 microseconds. It has no effect unless adaptive prefill
is enabled. An absent, malformed, or out-of-range value selects the relative
objective: twice the smoothed pure-generation step time, with a 1000 us floor.
The 100 us setting is a diagnostic that can force the admission path on a tiny
fixture. It is not the default policy or a recommended model setting.

Existing controls remain independent. In particular, preparation reuse does
not enable the MoE runtime or dynamic prefill provider by itself. Existing
graph-variant cache budgets also remain unchanged, including a previously
saved 128 MiB experiment. These patches do not silently enable that cache,
replace its configured limit, or change it to a universal default.

## Categorical-draw tail fusion

The sampler keeps the original candidate ordering, softmax, cumulative sum,
uniform RNG draw, and rounded F32 subtraction from the cumulative sum. Fusion
starts at the resulting difference vector. It replaces the subsequent
step/count/clamp/index/map fragment with one operation that counts strictly
positive differences and selects the same candidate.

The 1024-entry limit controls optimization eligibility; it does not truncate
the candidate set or change top-k, top-p, temperature, or grammar constraints.
Backend capability is checked using tensor metadata without allocating tensor
storage. Unsupported backends, layouts, types, and larger vectors keep the
original graph. The CPU sampler outside backend graphs is unchanged.

The automatic range is limited to standard CPU buffers because repeated
small-fixture complete-chain comparisons supported a gain there. CUDA timings
were mixed, including small slowdowns and differences near timing noise, so
CUDA remains an explicit comparison path. A faster isolated fragment does not
justify enabling a slower complete chain. These measurements do not establish
a model tokens-per-second improvement.

The fused operation writes the existing scalar token result. It adds no
persistent GPU workspace. Existing probability and candidate views remain
observable when requested. The separate existing selected-token-only path
continues to omit those views; fusion does not make them mandatory.

Correctness must cover the complete chain, not only the fused operation:
seeded draws, RNG continuity, reset, clones, state copies, transactional
multi-row acceptance and discard, graph rebinding, candidate order,
probabilities, constraints, and selected-token-only output contracts. Boundary,
nonmonotonic, nonfinite, and signed-zero inputs also need comparison against
the literal original graph. Finite outputs and selected token IDs must remain
exact; NaN classification is compared where payloads are backend-dependent.

## Bounded CPU preparation reuse for MoE

The existing dynamic provider computes small child graphs for routed expert
tiles. Preparation reuse retains at most four CPU work slots, matching the
provider's existing maximum number of queued jobs. It supports both the
token-major and expert-major builders. It does not enable expert-major
execution, change routing, select different matrix kernels, or change source
or accumulation precision.

Each slot retains one bounded CPU metadata arena, a graph template, and bounded
host ID-vector capacity. There is no additional GPU graph arena or device
workspace: every slot uses the provider's same existing workspace and buffer
layout. A cache hit requires matching shapes, strides, tensor types, resolved
buffer ownership, backend, workspace layout, tile geometry, and precision.
Structural changes rebuild metadata in the same CPU arena.

Current data pointers, source aliases, and view offsets are rebound only after
the previous consumer has completed. Asynchronous ID uploads retain their
source bytes until that fence succeeds. A failed drain keeps pending graphs
and vectors alive and blocks mutation; retry and teardown must respect the
same ownership rule. A workspace replacement invalidates templates after
draining pending work. Model unload drains before releasing metadata.

The stable CPU arena also bounds the number of first-node identities used as
CUDA graph keys. Child graph UIDs remain zero, so backend property checks still
see current pointers and view offsets. A stable key is not permission to reuse
stale tensor bindings.

The provider trace exposes cumulative `builds`, `reuses`, `invalidations`,
and first-node `key_changes`, plus retained `cpu_metadata`, `cpu_ids`, the
existing `gpu_workspace`, and `pending` work. Structural cache misses can be
legitimate even when first-node identity remains stable. A faster preparation
microbenchmark alone does not establish an inference benefit: a provider run
must demonstrate real reuse, completed work, identical outputs, and the same
GPU workspace in both controls.

## Adaptive prompt admission

The first policy changes when a prompt round is admitted, rather than changing
its token count, order, or configured batch sizes. After learning from completed
pure-generation steps and observing sustained mixed-work latency pressure, it
may defer the entire pending prompt round while already-generating sequences
take one ordinary step. Every deferral is followed by an ordinary admission
round. Prompts cannot be indefinitely delayed by consecutive policy deferrals.

The policy requires at least two clean pure-generation observations before it
reacts, uses hysteresis, and resets learning when the generation-row count
changes or a task or layout becomes unsupported. It uses already completed
synchronized observations and adds no GPU fence solely for telemetry. A single
request without competing generation retains ordinary admission.

Attaching a new slot's backend sampler can temporarily require a graph
reservation even though the workload remains supported. This transient state
uses ordinary prompt admission and contributes no timing sample. It resets
pressure and pending deferral state while preserving the learned pure-step
baseline if the already-generating row count is unchanged. A changed row count
or genuinely unsupported task or layout still discards that baseline. The
`reserve_admissions` counter distinguishes these ordinary reservation rounds
from policy deferrals.

Eligibility is intentionally narrow: continuous batching, ordinary completion
tasks, unified single-stream KV memory, and a supported single-device context.
Hybrid/recurrent or sliding-window memory, multimodal work, embeddings,
speculative or draft execution, MoE/RAM streaming providers, tensor overrides,
LoRA, grammar-constrained tasks, audio, and unsupported memory layouts retain
ordinary prompt admission. Expanding eligibility requires separate evidence;
the policy is not a universal scheduler for every model family.

During the entire eligible contention episode, including learning rounds,
transient reservation rounds, deferrals, and forced prompt rounds,
graph-variant lookup becomes reuse-only. An already cached compatible arena
may be used, but changed batch composition may not
admit a new secondary retained arena or prime normal cache-admission history.
Nested scopes and errors restore the previous mode. This protection also
persists through remaining pending work if a later task makes the episode
ineligible for further deferrals.

An ordinary round can grow the primary reservation before the policy has made
any deferral, for example when a new slot attaches its backend sampler. The
implementation records that ordinary primary and GPU growth separately,
resets latency pressure, and skips timing observations until reservations are
stable. It does not claim that ordinary prompt admission allocates nothing.

Once an episode has actually deferred a prompt round, subsequent primary or
primary GPU growth disables further adaptation, including growth during a
forced ordinary followup. New secondary retained storage remains forbidden
throughout the contention episode. This distinguishes ordinary setup from
memory growth that could follow an adaptive decision.

Trace-enabled OFF/ON correctness comparisons must check actual completed-work
peaks: the enabled primary, primary GPU, and secondary peaks must not exceed
the corresponding disabled control. The `ADAPTIVE_PREFILL_MEMORY` idle marker
reports cumulative actual `primary_peak_bytes`, `primary_gpu_peak_bytes`, and
`secondary_peak_bytes`. These peaks, rather than allocation deltas, are the
primary OFF/ON memory comparison.

The `ordinary_primary_growth_bytes` and `ordinary_primary_gpu_growth_bytes`
fields record the largest single-call ordinary growth deltas. The
`ordinary_primary_before_bytes` / `ordinary_primary_after_bytes` and GPU
counterparts describe the last ordinary growth call, which need not be the
call with the largest delta. They are diagnostic snapshots, not cumulative
peaks or evidence that ordinary setup never allocated memory.

These are runtime-managed buffer checks,
not an operating-system limit on all VRAM, driver allocations, KV storage,
model weights, or memory used by other programs. The configured graph-cache
budget primarily governs secondary retained arenas; it is not a universal
process-memory cap.

## Streaming slot identification

A related correction assigns the actual slot ID to partial and progress
results, consistently with final results. Native streaming responses that
include `id_slot` can therefore identify the slot correctly during concurrent
requests. The correction changes result metadata, not token generation or
sampling.

## Validation and activation gates

The initial tests use small synthetic fixtures and local tiny model files.
No large-model run, model download, context reduction, quantization change, or
saved-configuration change is needed to validate these implementation paths.

| Path | Correctness and memory evidence | Performance evidence |
| --- | --- | --- |
| Draw tail | CPU/CUDA literal-reference cases, complete sampling chains and RNG state, selected-token-only output, and actual graph-allocator bytes no greater than the original | Warmed isolated and complete graphs, eight ABBA cycles and 16 raw samples per control, reported separately |
| MoE preparation | Token/expert/mixed builders, F32/Q8 weights, exact outputs and metadata, pointer/offset rebinding, invalidation, failed-drain ownership, four-slot CPU bounds, and unchanged GPU workspace | CPU-only preparation benchmark plus actual dynamic-provider OFF/ON comparisons showing real reuse; trace disabled for timed runs |
| Adaptive admission | Deterministic fairness/learning tests, transient-reservation baseline preservation, actual backend graph reuse across alternating shapes, exact logits, separation of ordinary setup growth from post-deferral growth, enabled memory peaks no greater than the disabled control, and restored scope on errors | Concurrent stream latency, prompt completion/TTFT, aggregate throughput, and fairness with the default relative objective; isolated requests checked separately |

`test-dist-tail-fusion` and `test-moe-prepare` accept `--cuda` and `--bench`.
`test-adaptive-prefill` exercises the policy and scope ownership. Passing a
fixture-directory argument also exercises the graph contract with an actual
tiny backend fixture; `--cuda` selects CUDA for that fixture. Provider and API
tests must additionally exercise the real server paths; a helper test or log
line alone cannot prove activation.

A forced 100 us adaptive diagnostic demonstrates deferrals and restoration,
but its timing results must be separated from a default-relative-policy
comparison. Similarly, CPU graph-construction time is not model tokens per
second. Reports must retain controls, sample counts, raw timings, allocation
measurements, and rejected cases before any default is expanded. Improvements
on one component, device, or model family do not establish universal gains.

## Concurrent output and attention precision

The controlled CPU diagnostic uses identical recorded token histories in four
contexts. Changing only graph-cache admission is bit-exact; repeating the same
altered schedule is also bit-exact. Changing the physical interleaving of two
sequences is a different comparison and can change floating-point results.

With CPU Flash Attention, both F16 and F32 KV storage produced a maximum logit
difference of about 0.000605 and two near-tie argmax changes in the synthetic
fixture. Both failed the predeclared 0.00001 numerical gate. The graph casts
K/V to F16 for this attention path; its single-query and tiled CPU kernels also
use different query and accumulation precision. Increasing KV storage precision
alone therefore did not remove this sensitivity.

A separate control with F32 KV and Flash Attention disabled passed the same
unchanged gate: maximum absolute difference 0.0000001491, relative L2 error
below 0.000000216, and no argmax changes. The diagnostic changes no user setting.
It is a controlled mechanism test with reconstructed schedules, not an exact
replay of HTTP scheduling, a CUDA numerical bound, or a trained-model quality
measurement. Adaptive admission remains disabled by default. Natural concurrent
text equality is reported rather than promised; fixed-composition logits,
isolated natural output, forced concurrent output, cancellation and memory
contracts have separate checks.

## Optional VRAM in Wildgoose profiles

Wildgoose exposes per-model **Memória adicional de VRAM** controls for the
additional graph cache and optional DRY backend buffers. Disabling the graph
cache sends a zero budget to that model's runner and retains the chosen nonzero
budget for later reactivation. The estimate uses the effective enabled settings.
Existing choices, including a saved 128 MiB graph budget, are preserved.

These controls do not disable the automatic CPU draw-tail optimization or the
existing automatic top-p optimization. They do not change weights, logical
context, MTP settings, or memory quotas. Applying a changed memory option reloads
the affected model. The Go API accepts the bounded graph budget and explicit
DRY/overlap runner overrides; unrelated environment keys remain rejected.

The withdrawn CPU scratch experiment numbered 054 remains absent from the
patch stack. Its original baseline implementation is unchanged. These follow-up
patches do not reintroduce it.
