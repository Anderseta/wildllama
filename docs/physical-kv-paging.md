# Compatible physical KV paging

`WILDLLAMA_KV_CACHE_PAGING=1` opts into physical tail-page reclamation for the
existing CUDA VMM KV storage. It requires a positive
`WILDLLAMA_KV_CACHE_INITIAL_TOKENS`. An unset/zero initial commitment keeps its
existing policy; enabling paging does not silently choose a different value.
The initial physical floor, logical context and tensor virtual addresses remain
unchanged. No extra GPU cache or device workspace is reserved.

## What can be reclaimed

At a synchronized idle boundary, `llama_memory_trim_unused(ctx)` protects the
initial floor and the padded dense read span through the highest live physical
cell, including retained-prefix aliases. It also protects quantized allocation
padding for every tensor. Whole physical mappings beyond those ranges can be
released while the current conversation/prefix stays valid. The API returns the
physical bytes released; disabled, unsupported and unchanged cases return zero.

The optional mapping policy splits larger new mappings into segments of at
most 2 MiB, or one hardware minimum page when that is larger. Reservations with
more than 65,536 minimum pages retain the original mapping strategy instead of
creating more optional descriptors. Map metadata is CPU-side; every physical
allocation and recommit remains charged to the existing managed CUDA quota.
Driver/library allocations and other processes have their separate scopes.

Later decoding and state restore recommit the necessary pages before writing,
with the same zero-fill and allocation-denial contracts. Pending shifts are not
trimmed. Optional failure leaves still-mapped pages charged and forces future
growth checks to revalidate the lowered watermark.

Eligible plain unified single-stream caches can combine this storage with
multiple sequences and immutable shared-prefix aliases. The existing alias,
token and host metadata allowances remain in place. Compatible hybrid memory
delegates reclamation to its attention cache only; recurrent state is untouched.
Unknown/derived memories, SWA, transposed V, shared-owner caches and unsupported
backends retain their previous storage behavior.

## Important distinction

This is physical tail paging, not a general paged-attention allocator. Dense
attention kernels can load logically empty interior cells before applying their
mask. Those pages remain mapped, even when no token owns them. Decommitting them
would require a different read/write mapping contract across attention, shifts,
defragmentation and state operations. No interior-page zero-alias shortcut is
enabled here.

The separate experimental [page-table attention](paged-attention.md) feature
introduces that different read/write contract for admitted ordinary caches.
Enabling the established tail-paging option does not automatically enable it.

## Small validation targets

* `test-kv-trim-ranges`: opt-in parsing, bounded mapping segments/metadata,
  initial floor, dense hole protection and overflow-safe padding policy.
* `test-cuda-kv-growth --paging`: 16 MiB managed quota, exact retained bytes,
  released mappings, protected padding, stable address and zero-filled recommit.
* `test-hybrid-prefix --cuda-paging`: synthetic Qwen3.5 hybrid with a 128 MiB
  managed quota, a live retained prefix after a longer prompt, byte-exact
  attention/recurrent state and identical continuation logits after reclamation.

These are compatibility/resource tests, not trained-model throughput results.
The [validation record](benchmarks/bounded-kv-device-20261102.json) records exact
4 MiB and 8 MiB reclamation cases, byte-identical state/continuation, ownership
guards and the separate managed test quotas.
