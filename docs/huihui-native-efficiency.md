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

Patch `094` adds an experimental Q8 grouped-query attention pair kernel. It is
off by default; `WILDLLAMA_CUDA_GQA_Q8=1` explicitly opts in before backend
initialization. Eligibility is restricted to NVIDIA CUDA, 256-wide heads,
one query and sequence, Q8 K/V, an even query-to-KV head ratio and the supported
dense mask/layout. Sinks, softcap, ALiBi bias and unsupported shapes retain
the original kernel. The paired query heads keep independent accumulators and
write the canonical output and split-reduction rows; the existing merge remains
unchanged. GPU correctness and performance checks are required before enabling
this path by default.

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
The grouped-attention fixture also matched the original kernel exactly across
24 cases covering grouped heads, split-K, padded strides, masks and clipped KV.

The full patch stack was applied offline to a clean copy of the pinned native
archive; new patched files matched the compiled source. The API race tests
passed. These results do not alone establish inference throughput. Whole-model comparisons must
report context, sampling, warmup, memory and the exact payload identity; component
results alone do not establish a tokens-per-second improvement.

One sequential Windows/RTX 5060 Ti comparison used the same resident Huihui
Qwen35 27B IQ3_S weights, 65,536-token context capacity, Q8 KV, greedy sampling,
128 output tokens, one warmup and three warm requests per prompt length:

| Input tokens | Previous payload, median tokens/s | New default, median tokens/s |
| --- | ---: | ---: |
| 1,024 | 23.23 | 23.62 |
| 32,768 | 13.21 | 13.74 |

All generated text, selected token IDs, top-three IDs/order and reported log
probabilities matched exactly for the new default. Peak process working set was
about 267 MiB lower and peak private bytes about 127 MiB lower. Whole-device
VRAM differed by about 3 MiB; it includes unrelated display/driver allocations.
These short measurements establish no universal speedup or hard process-memory
limit, and small timing differences can vary between runs and hardware.

The optional grouped-attention path improved these timings further, but its
whole-model log probabilities differed reproducibly even though greedy text
matched. The fixed-split component fixture is therefore insufficient evidence
of whole-model numerical equivalence. This path remains experimental and off;
it is excluded from the default results above and needs further investigation.

Builds and model tests in this workspace use owned process trees with RAM,
commit-headroom and VRAM guards. Test outputs remain separate from the installed
runtime, user profiles and model store.
