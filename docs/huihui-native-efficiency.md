# Format-aware native efficiency

The changes in this round keep model weights, quantization, context, sampling
and memory allowances independent of kernel selection. They are motivated by a
resident Qwen35 dense/hybrid IQ3_S model, but eligibility is determined from
actual tensor types, shapes, layout and backend capabilities.

## Shared quantized activations

Patch `092` admits IQ3_S to the existing Q8_1 activation-reuse path. IQ3_S's
importance matrix determines weight quantization; its MMVQ activation layout
is the same Q8_1 layout used by the previously supported formats. Sharing does
not requantize stored weights or convert Q8 models to IQ3_S.

IQ3_S activation reuse defaults on independently of the legacy
`GGML_CUDA_DENSE_OPT` switch. Set `GGML_CUDA_IQ3_INPUT_REUSE=0` before backend
initialization to disable this addition for a comparison. This default does not
enable the legacy Q8/K-quant optimizations or fused operations: those keep their
existing control and behavior.

The existing value-version, alias, stream, workspace and managed CUDA quota
guards still apply. Reuse requires compatible consumers of the same input
value; unsupported operations and unavailable workspace use ordinary dispatch.
This is a format-specific extension, not a speedup claim for every model.

Explicit CUDA calibration can also consider IQ3_S and the 17,408-wide feed-forward
dimension. No unmeasured launch heuristic or calibration policy is enabled by
this extension. The existing finite, bitwise comparison, repeated-win checks,
trusted policy identity and memory bounds remain authoritative. A large fixture
that cannot fit the calibration budget is rejected instead of increasing that
budget; ordinary runtime dispatch remains available.

## Immutable recurrent checkpoints

Completed host checkpoints may share immutable serialized bytes across prompt
copies. Updating one owner must obtain independent mutable storage first;
clearing a copy releases its ownership without changing another copy. The
serialized target, draft and speculative state remains byte-exact.

Logical per-entry cache charging remains conservative rather than granting a
larger cache allowance when entries share bytes. Slot checkpoint RAM admission
counts each shared payload allocation's capacity once, including retained
capacity; bookkeeping allocation failure fails closed. Allocator metadata,
model weights and other process allocations still exist, so this allowance is
not a hard process-RSS limit. This optimization adds no device buffer.

Synchronous spill reads shared bytes without detaching them and releases only
its own references after a complete save. Small shared checkpoints retain
asynchronous prewriting when the existing writer ticket, cache quota and
physical RAM headroom admit a bounded copy. Oversized checkpoints or low RAM
skip that optional copy; restore and ordinary spill remain available. Partial
allocation failure preserves the other owners and their original bytes.

## Attention and sampler boundaries

Patches `094` and `095` provide guarded Q8 grouped-query attention. The final
implementation preserves the legacy KV partitions and uses two independent
four-warp head groups, with specialization confined to conditions checked by
the host. Automatic selection is limited to SM120, 24 query heads and four KV
heads, 256-wide Q8 K/V, one query/sequence and the supported dense F16 mask and
strides. Sinks, softcap, ALiBi bias and unsupported shapes retain the original
kernel. `WILDLLAMA_CUDA_GQA_Q8=0` disables the addition; `1` explicitly opts in
on other eligible NVIDIA layouts. See [resident CUDA kernels](resident-cuda-kernels.md)
for arithmetic, activation and validation details, including the separately
bounded IQ3_S two-row matrix-vector path in patch `096`.

Thinking-aware backend sampling cannot drop the reasoning-budget object merely
because its limit is effectively unbounded: that object also controls lazy
grammar activation. Backend attachment and speculative rows must observe phase
transitions correctly. This round preserves those guards rather than changing
the sampling distribution or tool behavior.

## Validation

Ten focused CPU tests passed: state pager, state writeback, hybrid prefix,
CUDA tuning policy, checkpoint-buffer ownership, GQA eligibility/row mapping,
prompt-state transactions, and MTP state, device-state and carry-state handling.
The prefix-cache integration fixture also passed unchanged logical charging,
shared-owner spill, eight exact warm/cold cycles, small shared async writeback,
low-headroom rejection, and missing/corrupt backing recovery. The standalone
ownership test verifies that copying a checkpoint does not allocate its payload
and that failed detachment preserves both owners.

CUDA projection checks passed for eleven quantizations, single/multiple query
tokens and mixed consumers. The output dumps were byte-identical with reuse
disabled, the IQ3-only default and the legacy opt-in. Counters confirmed that
the IQ3 default actually ran without activating pure Q8/K reuse; all three
stayed below the 64 MiB test quota and had the same managed allocation peak.
The grouped-attention production-planner fixture matched the original kernel
bit for bit across 108 cases, including several grouped-head geometries,
short/long KV views, three input patterns and padded strides. Negative controls
reproduced the old partition-selection error. Nine trained-weight IQ3 row-pair
cases also matched bit for bit. The square-range fixture uses a 5,120-row prefix
of an actual 10,240-row QKV tensor; it does not establish a measured benefit for
that full QKV shape, which remains on the original path.

The full patch stack was applied offline to a clean copy of the pinned native
archive; new patched files matched the compiled source. The API race tests
passed. The final whole-model comparison used Windows/RTX 5060 Ti 16 GiB,
resident Huihui Qwen35 27B IQ3_S weights, 65,536-token context capacity, Q8 KV,
greedy sampling and 128 output tokens. Each of baseline, attention-only and
combined automatic settings completed eight requests: one warmup and three
measured warm requests at each input length. Medians use native completion
decode timings, excluding WildGoose and model startup/loading.

| Input tokens | Previous CUDA payload, median tokens/s | Attention | Combined automatic |
| --- | ---: | ---: | ---: |
| 1,024 | 26.946 | 27.478 (+1.98%) | 27.448 (+1.86%) |
| 32,768 | 16.530 | 22.647 (+37.00%) | 22.532 (+36.31%) |

Model/profile and prompt identities matched. The candidate CUDA payload was
distinct while eight common native files were unchanged. All accepted cases
had exactly identical content, selected token IDs and reported selected and
top-three token IDs/order/log probabilities. The combined result is less than
1% from attention-only at either length, so it establishes no separate
whole-model throughput gain for `096`. Its small component results do not
justify claiming additional model tokens/s.

One attention timing window contained paging and was discarded; its repeated,
accepted window and every other retained timed window passed the paging checks.
This does not claim the entire machine was paging-free outside those windows.
The suite measured performance and recorded probability equivalence, without
separate functional/tool-call cases. These short measurements establish no
universal speedup, complete vocabulary-logit equivalence or hard process-memory
limit. See [resident CUDA kernels](resident-cuda-kernels.md) for activation
controls, the corrected FP32 partition/reduction order and component results.

Builds and model tests in this workspace use owned process trees with RAM,
commit-headroom and VRAM guards. Test outputs remain separate from the installed
runtime, user profiles and model store.
