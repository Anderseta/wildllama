# Page-table attention and sparse physical KV

`WILDLLAMA_PAGED_ATTENTION=1` opts into the page-table attention implementation.
It is independent of `WILDLLAMA_KV_CACHE_PAGING`, which retains its established
dense-attention tail-reclamation behavior. Both remain disabled unless explicitly
selected. Existing profiles, logical context, model weights and sampling settings
are preserved.

## Storage and attention

An admitted unified cache groups physical KV cells into 32-token pages. A compact,
sorted block table lists the live physical pages. Attention reads those pages
directly with the original F16, F32 or Q8_0 storage; it does not gather or
requantize the KV history into another context-sized allocation. Sequence and
causal visibility are expressed in the same logical block order. Shared prefix
cells remain referenced until their last sequence owner releases them. New
writes first use free cells in active partial pages and then empty pages.

The CPU implementation uses bounded per-thread row scratch. The CUDA paged kernel
uses bounded registers/shared memory and no context-sized global workspace.
Table and allocator metadata use host memory; uploading the table and mask still
uses the ordinary graph allocator and existing memory accounting.

When live pages form a contiguous, 256-row-aligned span, compatible CUDA F16/Q8_0
consumers can reuse the existing Flash Attention kernel over a cropped view.
Admission requires backend support and no extra result allocation beyond the
ordinary output. Graph reuse checks the exact physical span; a changed layout
rebuilds the graph instead of reusing stale cropped addresses. Fragmented or
incompatible spans use the direct page-table kernel.

For physical VRAM savings, set a positive `kv_cache_initial_tokens` as well. At a
synchronized idle boundary, the cache protects every live page, the initial floor
and quantized allocation padding. It can release unused interior CUDA VMM
mappings after every cache layer has proved that its attention reads use the
page table. Actual CUDA mapping granularity can be coarser than a 32-token page;
an overlapping live range retains its whole mapping. An unset initial commitment
keeps conventional physical allocation even when attention uses page tables.

All new physical commitments and recommitments remain inside the existing managed
CUDA quota. Allocation failure returns an error before new sequence cells are
published. This does not impose a limit on other applications or driver-owned
allocations. It never spills KV to SSD or changes the operating system's paging
policy.

## Compatibility and fallback

Admission checks the actual per-layer storage and backend support before enabling
the allocator. The initial implementation supports ordinary unified, single-stream
attention caches, multiple sequence IDs, grouped-query attention and F16/F32/Q8_0
K/V on CPU and compatible NVIDIA CUDA devices, with Flash Attention enabled.
An exact ordinary hybrid cache can use the same attention pages while keeping
its recurrent state on the established path. Admission waits until the complete
hybrid owner and its attention/recurrent child types can be checked. It preserves
the original path for unsupported backends, transposed V, SWA,
indexed/compressed attention, shared-owner
caches, tensor placement overrides, derived/indexed hybrid memories and
multi-device layouts.

An unexpected dense attention consumer inhibits sparse reclamation and populates
its original physical-order mask. Dense backing is committed only for actual
execution, inside the same quota; worst-case graph reservation does not allocate
the full virtual KV capacity. Context shifts similarly recommit the range read
by the existing maintenance kernel. State snapshots keep their existing format
and serialize live cells; restore recommits destination pages before writing.
A legacy shared-cache view can retain ordinary initial growth when its source
is empty and has never started paged work. Otherwise it first requires the
source's complete dense backing; admission can fail within the existing quota
rather than exposing unmapped pages. Sharing preserves the source's existing
cells instead of resizing/resetting them. Hybrid allocation failure stops before
applying recurrent state, and speculative planning restores its recurrent hints.

This is experimental. Floating-point reduction order can differ from conventional
attention. Small numerical/resource regression tests do not establish higher
trained-model throughput or parity with SGLang. Its main purpose is safe sparse
KV addressing and reclamation under memory pressure, with a correct conventional
fallback.

## Focused validation

The validation targets use synthetic tensors or tiny generated models:

- `test-paged-attn` and `test-cuda-paged-attn`: independent attention reference,
  mixed KV types, strided storage, GQA, masks, multiple streams, sinks, soft caps,
  partial lengths and invalid metadata.
- `test-paged-kv-pool`: physical allocation, shared ownership and transaction
  rollback without device memory.
- `test-kv-page-ranges`: live islands, initial floor, quantized padding, bounds,
  overflow and bounded range metadata.
- `test-cuda-kv-growth --paged-attention`: small managed CUDA quota, interior
  reclamation, preserved live bytes, residency queries and zero-filled recommit.
- `test-kv-paged`: end-to-end synthetic model cache, branching, state and
  continuation behavior; its optional CUDA mode has a separate small quota.
- `test-kv-paged-hybrid`: an ordinary synthetic Qwen3.5 hybrid, attention/RNN
  state integrity, retained-prefix and full/partial restores, and allocation
  denial without advancing recurrent state.

The [small validation record](benchmarks/paged-kv-attention-20261104.json) includes
16 CPU checks, six CUDA checks, exact state/resource contracts and numerical
comparison scopes. No pretrained model was loaded or downloaded for these checks.
