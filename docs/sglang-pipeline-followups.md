# Bounded graph reuse, indexed KV operations, and output overlap

Patches 049–053 extend the [dense runtime defaults](dense-default-runtime.md).
They adapt execution ideas to the existing GGML engine and preserve model
weights, quantization, requested logical context, and existing memory limits.
They do not establish performance parity with SGLang.

## Defaults and explicit experiments

| Change | Activation | Memory contract |
| --- | --- | --- |
| Indexed removal of a sequence's KV cells | Automatic | Uses the existing position index; no new GPU tensors. |
| Correct adaptive-p checkpoints and smaller derived scratch copies | Automatic | Copies logical RNG/EMA state and one pending probability. |
| Broader retained-prefix sharing | Existing explicit prefix-retention setting | At most 32 aliases, existing physical-cell allowance, bounded extra CPU metadata. |
| Repeated prefill/verification arenas | WILDLLAMA_GRAPH_VARIANTS_MIB greater than zero | Extra metadata and compute storage share the per-context budget. |
| DRY backend penalties | WILDLLAMA_DRY_BACKEND=1 | Bounded sparse inputs and conservative graph admission; existing quotas remain authoritative. |
| Streaming-output/decode overlap | WILDLLAMA_DECODE_OVERLAP=1 | At most one ordinary forward ahead; uses existing compute and KV reservations. |

Library defaults for these runtime controls remain disabled. The WildGoose
experiment can explicitly select a 128 MiB graph-arena budget.

## Indexed KV work and retained prefixes

Removing a nonnegative sequence ID now walks that sequence's existing position
index over the requested range. It preserves the original physical free-cell
head, other sequence memberships, position metadata, and range endpoints.
The all-sequence removal path retains its original implementation.

Eligible retained prefixes also create and restore aliases through indexed
positions. Identical physical mappings can share a retained handle. The registry
supports at most 32 aliases; admission counts unique protected physical cells,
so overlapping prefixes do not multiply the physical-cell allowance.

The existing configured retained-token bound and quarter-capacity physical
reserve remain in place. Additional alias bookkeeping uses a conservative
1 MiB CPU metadata admission allowance. The charge includes implementation
allowances for set nodes/control blocks; it is not a process-RSS limit.
Existing four-entry/logical-budget admissions retain their previous allowance.
Pressure releases cold aliases in a bounded retry before retaining the normal
invalidation/serialization fallback.

Plain eligible attention storage remains required. The existing hybrid wrapper
still owns and charges each recurrent snapshot independently. This is not a
general paged-attention allocator for every SWA, spatial, or derived memory.

## Repeated prefill and verification arenas

The optional arena cache now considers supported batches of 9–512 tokens,
in addition to the existing short-decode path. Larger shapes require two
observations in a bounded eight-key admission history. Admission history is a
coarse filter; every actual hit must still pass the complete graph/input,
memory, sampler-binding, and placement reuse contract.

Up to four secondary arenas share the existing budget. Metadata capacity uses
power-of-two buckets while actual token shapes and computation remain unchanged.
Fixed metadata slots avoid generating new CUDA graph keys on every eviction.
Failed optional admission uses the primary scheduler and suppresses repeated
expensive retries for the same recent shape.

The budget is **per inference context**, not a server-wide VRAM cap.
Target/draft inference can have two contexts: 128 MiB each permits up to
256 MiB of combined additional arena budgets. Multiple model runners also have
independent budgets. Metadata and CPU compute buffers can consume RAM; eligible
CUDA compute buffers consume VRAM. Weights, active KV capacity, the primary
scheduler, driver allocations, and other library allocations are separate.
The managed CUDA allocation quota remains independent and authoritative for the
allocations it tracks. Optional arenas do not reduce context to fit.

## DRY backend and sampler checkpoints

The optional DRY backend derives the existing repetition penalties from
accepted CPU history using the original Z-algorithm and power calculation.
It uploads bounded sparse token IDs/penalties and applies subtraction through
supported backend operators, avoiding full-logit downloads in eligible chains.

Admission supports one output, at most 65,536 sparse entries, and a conservative
8 MiB bound for added graph tensors/metadata, including possible vocabulary-sized
intermediates. Host staging is bounded at 512 KiB. Actual CUDA allocations
remain subject to the existing quota and available memory. Unsupported operators,
oversized plans, and denied admission retain the CPU suffix.

Enabling a newly complete backend chain can select tokens using a different RNG
algorithm from the previous CPU suffix. The same seed need not produce the old
sequence. Consequently this extension requires exactly WILDLLAMA_DRY_BACKEND=1;
existing backend profiles keep their previous route. The raw DRY penalty
calculation and sampling distribution require separate numeric/contract validation.
Existing host and backend greedy selectors can also resolve tied maxima
differently. This extension preserves each selector's existing tie rule.

Final constrained validation accepts the existing contiguous top-k row layout
through a one-dimensional metadata view when needed. This preserves values and
candidate indexing while avoiding a vector-contract assertion for complete
grammar-constrained stochastic chains. Existing vector inputs are reused.

Independently, adaptive-p clones now retain the advanced RNG, actual seed, EMA,
and the pending chosen probability. DRY clones retain history while rebuilding
derived workspaces when needed. Reset with an explicit seed remains repeatable;
independent random-seed resets deliberately choose fresh seeds.

## Streaming response work overlapped with a normal forward

The experimental output-overlap path starts one ordinary forward using an
already accepted token while CPU partial-response handling continues.
This is not an autonomous GPU token-feedback loop: token selection still has
its host synchronization boundary. It introduces no placeholder token or
device-to-device token relay.

Initial admission is deliberately narrow: one streaming text slot, exact plain
KV storage, supported CUDA placement, complete compact greedy backend sampling,
and no MTP, history penalties, grammar, detailed probabilities, media, adapters,
indentation, or time-based generation controller. CPU token embedding lookup
can remain when the CUDA trunk otherwise qualifies.
The existing inactive time-controller values (negative or zero) remain eligible;
positive time limits retain the ordinary path.

Known EOS/stop/limit decisions happen before queuing the next forward. A ticket
owns at most one pending step. Cancellation, errors, reset, and unload drain it
before trimming its KV/prompt state. Consuming a completed ticket skips duplicate
batch admission. Incompatible requests retain the existing path.

The CPU window is limited because HTTP JSON serialization already runs on
another thread. This feature remains opt-in until relevant end-to-end
measurements establish a benefit.

## Native capability discovery and validation

The --wildllama-runtime-features flag returns schema-1 JSON naming the supported
runtime environment keys, then exits without loading a model. WildGoose reads
this declaration before enabling the corresponding fields. It also refreshes
cached capabilities when companion DLLs change.

The [validation record](benchmarks/sglang-pipeline-20261031.json) separates native
CPU/CUDA tests, exact default API comparisons, optional DRY/overlap contracts,
component timings, and measured memory. Fixtures are local synthetic models up
to 64 MiB. Their timings do not predict trained-Qwen tokens/s or language quality.

## Small-fixture measurements

The Windows RTX 5060 Ti measurements use the same final binary and a local
approximately 14 MiB synthetic Llama fixture. They compare individual feature controls,
not overall WildLlama throughput against another engine.

With eight warm ABBA cycles, eligible GPU output overlap changed native decode
throughput from 1,625.9 to 2,148.8 tokens/s (+32.2%). The 128-token streaming
response took 86.07 versus 64.16 ms. First nonempty content arrived at a median
4.751 versus 6.858 ms, about a 2.1 ms latency cost. This throughput/latency tradeoff
and the narrow eligibility rules are why the option remains disabled by default.
CPU controls did not activate overlap; their timing variation is control noise.
The first health-ready completion is reported separately, excludes model loading,
and can follow native capability probes. It is not fully cold CUDA startup.

The graph-cache experiment used four fresh processes in 0/128/128/0 MiB order,
one 2,048-token context per process, and eight cycles with quiet logging:

| Prefill shape | Cache 0, median wall time | Cache 128, median wall time |
| --- | --- | --- |
| 32 tokens | 2.084 ms | 0.485 ms |
| 128 tokens | 2.352 ms | 0.769 ms |
| 512 tokens | 3.000 ms | 1.754 ms |

All four trials produced bit-exact logits. The enabled cache reused arenas 100
times and reached four additional arenas: 8,878,720 bytes of compute storage
plus 1,673,024 bytes of host metadata, totaling 10,551,744 bytes (10.06 MiB),
within the selected 128 MiB per-context allowance. The observed warmed
device-global memory difference was 6 MiB; WDDM did not expose per-process
memory. That global sample cannot establish a process VRAM increase or a total
VRAM cap. Model weights, KV, the primary arena, driver allocations, and other
applications have separate accounting.

Final validation includes 38 CPU and 12 CUDA tests, seven small architecture/
quantization families in both CPU and CUDA graph/prefill runs, 300 default API
cases per payload, 127 DRY requests on each device, and 15 ordinary overlap SSE
cases plus cancellation/recovery controls. The GPU overlap proof recorded 212
queued steps, 211 consumed steps, and one cancelled/discarded step.
