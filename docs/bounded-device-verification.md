# Bounded CUDA speculative verification

The [small validation record](benchmarks/bounded-kv-device-20261102.json)
separates actual MTP/candidate correctness, optional-allocation pressure,
physical KV compatibility and model-free driver overhead measurements.

`WILDLLAMA_VERIFY_DEVICE_MIB` selects an optional per-target-context scratch
allowance from 0 through 64 MiB. Zero or an unset value keeps the existing path.
It does not enlarge the managed CUDA quota or change the model context,
quantization, draft depth or sampling settings. WildGoose exposes this allowance
in the model profile's additional-VRAM controls.

## Eligible execution

The first implementation accepts explicitly attached, complete greedy backend
sampler chains with supported stateless filters. It requires one CUDA backend,
one sequence, one verification microbatch and at most 512 draft tokens plus the
target correction. Grammar, requested probabilities, reasoning controllers,
stochastic selectors, weight-streaming providers and unsupported layouts retain
the original implementation. A positive allowance does not change temperature
or automatically enable backend sampling.
Positive `WILDLLAMA_KV_CACHE_INITIAL_TOKENS` configurations require an explicit
cooperative resource link. Common's speculative target/draft initialization
installs that link for known plain KV, recurrent and hybrid memory contexts.
An unlinked context, shared attention storage or an unknown/derived memory
layout keeps the original verifier. The user's initial-KV setting is preserved.

The model's original CUDA sampler still selects each target token and retains
its existing tie rules and floating-point behavior. A CUDA kernel then compares
those selected device tokens with the draft, applies the same end-of-generation
boundary rule, and determines the accepted prefix and target correction. The
decision does not read logits or probabilities on the CPU and consumes no RNG.
The kernel returns a bounded token/count report. Common greedy chains usually
already compact each row to scalar logits/candidates, so their possible benefit
is fewer per-row driver-copy calls, not elimination of vocabulary-sized traffic.
Small draft batches may regress because uploads, the kernel and final public
candidate materialization add work. A positive allowance remains an experiment,
not a throughput guarantee.

Host sampler acceptance, stop-string handling, response formatting and KV/draft
rollback keep their existing ownership. This is GPU verification of an eligible
speculative batch, not an autonomous device-side generation loop or general
stochastic rejection sampling.

## Output and lifetime contracts

The original graph output tensors are retained until the next logical decode;
no vocabulary-sized device copies are introduced. Their ordinary host transfers
are deferred only for the proven complete single-microbatch layout. The final
accepted row is materialized for the common sampler's public candidate view.
Explicit context getters materialize other rows on demand, preserving selected
IDs, candidate order, logits, probabilities and counts.

A new valid decode invalidates the previous output view as usual. Empty or
invalid-input calls preserve it. Graph reservation and sampler replacement
materialize deferred rows before destroying their source bindings. Reordered
or split microbatches keep the original bulk-transfer path. Teardown drains
pending work before releasing scratch or model/backend storage.

## Memory scopes and fallback

All device scratch lives in one ordinary tracked CUDA buffer. The admission
includes its aligned end-of-generation flags, selected-row pointer descriptors,
draft IDs and result capacity. The end-of-generation table uses one byte per
vocabulary token; pointer and token descriptors have fixed 513-row bounds.
The context lazily allocates only the needed bytes, rather than reserving the
whole chosen allowance. A denied budget or managed CUDA allocation uses the
original path without reducing context or overflowing either allowance.
Mandatory graph re-reservation releases retained verifier scratch first, after
materializing deferred public outputs and draining queued work. A later eligible
completed graph can recreate scratch only if the remaining quota admits it.
Known memory contexts also preflight each microbatch immediately before its
memory apply. The cache uses the same required-cell calculation for preflight
and commit. A required growth boundary, unknown memory operation or state
restore first materializes deferred output, drains its work and releases
optional scratch. Stable ordinary tokens retain the existing bank. Reclamation
can be conservative at a cell-growth boundary even if its physical CUDA pages
were already mapped; this avoids duplicating backend allocation policy.

An explicitly linked draft can release its target's bank before the draft's
own growth, graph reservation or state restore. There is at most one live child
per target. The registration uses weak lifetime tokens and expires on teardown;
it does not install a global allocator callback. This contract requires the
existing nonconcurrent, ordered target/draft worker. Independent concurrent
contexts and unregistered users of shared device quota are outside this narrow
cooperation contract. Optional scratch is not a permanently reserved priority
allocation, and this mechanism does not promise arbitrary cross-context memory
preemption or a process-wide VRAM ceiling.
An allocation denial or failed host-metadata allocation delays another attempt
for 64 eligible observations. Unsupported work does not consume this cooldown.
The next eligible observation can retry, so transient pressure neither rebuilds
the vocabulary table every token nor disables the feature permanently.

The existing CUDA quota covers tracked allocations, including this scratch;
driver/library allocations and other processes remain outside its scope. The
MoE host-weight RAM allowance is unchanged and is not a process-RSS limit.

## Validation

`test-device-verification` checks budget parsing and conservative chain admission
without a GPU. Its `--cuda` mode checks mismatch/full-acceptance/EOG boundaries,
invalid selected-token sentinels, row bounds and scratch admission. A nonzero
synthetic Qwen/MTP fixture below 4 MiB also executes drafting, target verification,
the common-sampler fast path and public getters under a 96 MiB managed CUDA test
quota. The reference and candidate run sequentially to avoid retaining four
inference contexts; the test quota includes the two existing cuBLAS workspaces,
whereas the verifier's own allowance is 1 MiB. The fixture compares accepted tokens and candidate/logit/probability bits
with the original path and reports actual device-verifier hits and transfer
accounting. These contracts do not establish trained-model tokens/s or universal
model support; the feature remains an explicit experiment.

The separate `--growth` fixture uses a wider, untrained synthetic MTP model and
an 8192-token logical context under a 128 MiB managed CUDA test cap. It exercises
target and linked-child growth, stable bank reuse, child state restore,
shared-owner fallback and parent-before-child destruction. Its baseline and
candidate execute sequentially; it compares tokens and serialized target/draft
state without changing context, sampling, the 256-cell initial setting or the
verifier's 1 MiB allowance. The test cap also covers ordinary model, graph and
cuBLAS allocations; it is not additional verifier memory.

The separate `--bench` mode measures model-free driver-copy and prefix-decision
work. It compares per-row pinned D2H copies plus the host decision with the
device API, its descriptor uploads/kernel/report and final public-view copies.
Both controls verify the same full-acceptance token report. Three four-byte
arrays match the actual greedy fixture; a separate four-array control is also
reported. Scalar payload proxies exclude neural inference, sampler arithmetic
and the accepted-token bookkeeping common to both paths.

Before batched host-output transfers were added, one Windows RTX 5060 Ti run,
while the device was shared, recorded these
three-array median microseconds per round after warmup, using four ABBA cycles
with 16 iterations per block:

| Draft rows | Existing row-copy/host decision | Device verifier/public view |
| --- | ---: | ---: |
| 2 | 222.84 | 194.85 |
| 8 | 720.93 | 191.64 |
| 16 | 1,280.21 | 194.21 |
| 64 | 4,758.92 | 191.76 |

This isolates fewer driver calls, not inference throughput. In particular, the
smallest case has a modest absolute difference and is sensitive to shared-GPU
scheduling and platform overhead. Other backends, batch shapes and real models
can behave differently; these measurements do not justify enabling the feature
by default or predict a model tokens-per-second gain. The feature remains off
unless explicitly selected. The regular greedy fixture's 48 deferred bytes
include its one-row prefill plus three verification rows, at 12 bytes per row;
they must not be described as eliminated vocabulary-sized transfers.
The newer batched host-output path can also reduce reference transfer calls
where its tensor layout is eligible. The historical numbers above do not
compare against that newer baseline or establish an additional verifier gain
on top of it.
