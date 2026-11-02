# Runtime pipeline efficiency

Patches 079–082 reduce response serialization, repeated output fences and
weight-cache bookkeeping. They preserve model weights, context, sampling,
profiles and existing resource limits. No additional device bank, pinned upload
ring or KV reservation is introduced. Compatible paths apply automatically;
unsupported consumers and backends retain their existing behavior.

The [small validation record](benchmarks/runtime-pipeline-20261102.json) covers
12 CPU targets, two small CUDA modes, Go consumer tests and HTTP contracts.
All fixtures are synthetic and untrained. The GPU was shared; Linux, HIP and
Metal were not runtime-tested.

## Response selection

[Demand-driven response fields](prompt-response-efficiency.md) use the native
server's existing field-selection mechanism. The Go runner requests every field
its completion decoder consumes. Native default and explicitly requested prompt
echoes are preserved; compatibility formats avoid an unused prompt conversion.
Nested selection borrows immutable parent values and copies only selected leaves.

With a 1,536-token synthetic prompt, a streaming response transferred 12,430 bytes
with the default native fields and 1,785 bytes with the selected fields. Content,
probabilities, stop metadata and token counts matched. Explicit prompt echoes
and nested-path behavior were preserved. Six before/after HTTP cases per payload
also preserved responses, including ordinary, stochastic, probability and
concurrent grammar cases.

## Neural outputs

[Scoped neural-output views](neural-output-views.md) wait once for one completed
draft forward. A strict builtin-sampler guard permits subsequent host-only reads;
custom samplers and public getters keep their original synchronization contract.
Lazy device hidden state still materializes when it is actually requested.

Three samples and two hidden reads reduced context completion calls from five
to one. With CUDA and CPU backends attached, the observed backend callbacks were
ten versus two. Tokens, candidate bits/order, hidden state and subsequent random
draws matched. The CUDA fixture peaked at 36,759,680 tracked bytes under its
96 MiB managed cap. Component savings were microseconds, rather than a measured
large-model throughput increase.

## Conventional MoE uploads and RAM metadata

[Normal-cache uploads](moe-normal-cache-uploads.md) can use the existing batched
transfer path when their device advertises asynchronous operations and host
buffers. Capability checks happen once per device. Unsupported devices bypass
batch descriptor preparation and keep the previous prefetch/copy sequence.
Admission denial and I/O failure preserve existing quota and rollback rules.

The small CUDA transfer fixture reduced explicit H2D synchronization from nine
to one, keeping identical bytes, untouched bank slots, cold read counts and peak
weight RAM. Its warm copy component measured 1.2916 versus 0.3175 ms. A real tiny
normal-cache MoE decode compared 6,216 logits with a maximum absolute difference
of zero and exercised 57 cache misses. This is not an end-to-end trained-model
tokens-per-second measurement. The forced CPU helper verifies correctness only;
normal GPU-cache devices and capability admission determine production use.

[RAM queue-node recycling](moe-ram-queue-nodes.md) preserves SLRU selection and
iterator order while reusing list nodes across evictions. Active, in-flight and
recycled page nodes share the same bounded total; allocation remains lazy.
Physical-read failures drain I/O and return both committed and planning-only
nodes before retry. The existing two-page weight-cache fixture recovered through
eight repeated failures in pooled and nonpooled modes.

After warmup, the allocator fixture performed 1,024 cycles over 128 nodes:
the original queue allocated 131,072 nodes, while the recycler allocated none.
This reduces heap churn; it does not enlarge the physical weight cache or claim
a bound on total process RAM. Fixed container metadata remains separate.
