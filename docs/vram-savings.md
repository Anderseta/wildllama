# Optional VRAM savings

These runner switches trade allocation reuse or attention-kernel selection for
less device memory. Each is independent, disabled unless explicitly set to `1`,
and can be saved in a model's `runner_env`. Wildgoose groups them under
**Model profile → VRAM → Economia de VRAM**, with a gain/cost description and
additional help for each switch. Save and apply the profile to reload its runner.

They do not change model weights, stored KV precision, logical context, sampling,
draft depth, or the configured managed CUDA quota. Unsupported layouts retain
the conventional path. A runner advertising a switch is not a promise that
every model, backend, adapter, or graph can use its optimized path.

| Runner environment | Memory behavior | Possible cost |
| --- | --- | --- |
| `WILDLLAMA_COMPUTE_GROWTH` | Reserve attention-related compute graphs against a growing physical-row high-water mark instead of the complete logical context. | Synchronized scheduler/graph reconstruction at growth boundaries; long histories eventually need larger arenas. |
| `WILDLLAMA_COMPACT_ATTN_MASK` | Compatible CUDA Flash Attention consumes compact causal/sequence metadata instead of a complete dense mask. | Additional visibility calculations inside attention; throughput depends on the kernel and shape. |
| `WILDLLAMA_CUDA_POOL_TRIM` | Return eligible unused CUDA scratch allocations after a synchronized phase, retaining a warm allowance and release hysteresis. | Later allocations or graph captures may have to be rebuilt. |
| `WILDLLAMA_PAGED_ATTENTION` | Read compatible target KV pages directly and permit safe sparse physical reclamation. | The general page-table kernel can be slower than mature contiguous Flash Attention. |
| `WILDLLAMA_PAGED_DRAFT` | Extend page-table admission to a separately proved attention-only MTP cache. | The auxiliary attention path may reduce MTP's throughput benefit. |
| `WILDLLAMA_KV_CACHE_PAGING` | Return eligible unused physical tail mappings of the existing dense KV cache. | Future growth must recommit those mappings. |

Explicit `0` overrides an inherited enabled switch. Removing an override restores
the existing environment/default behavior. None of these switches enables MTP,
selects another model, reduces the context, or chooses a new physical KV floor.
The physical paging switches require a positive `kv_cache_initial_tokens` for
physical reclamation. Compute growth is independent: when no physical KV floor
is configured, its initial compute reservation uses 8192 rows or the microbatch,
whichever is larger, bounded by the logical cache capacity.

## Compute growth and correctness

Admission is limited to proved ordinary causal Flash Attention memories on
compatible CPU/CUDA layouts. Exact ordinary hybrid memories delegate to their
attention cache; recurrent state keeps its existing representation. Indexed,
SWA, shared-owner, overridden-placement, training, pipeline and unsupported
multi-device layouts retain conventional reservation.

The high-water mark covers existing live physical rows and all planned writes,
including holes and restored histories. It grows geometrically with allocation
padding. The scheduler reserves the larger graph before applying the prepared
KV/recurrent batch. Allocation denial must return an error without advancing
those states; a later retry may reserve again. This is a starting allocation
policy, not a promise that an arbitrarily long conversation fits a small quota.

## Compact masks and scratch trimming

Compact attention is admitted only after the existing sequence/position proof
and backend capability checks. General additive biases and unproved layouts keep
their original dense masks. Page-table attention retains its own logical-order
mask; enabling both switches does not imply two independent memory savings.

CUDA scratch trimming acts on returned temporary allocations, not model weights,
KV cells or live recurrent states. It uses the existing synchronization boundary,
completes owned streams before releasing backing memory, and invalidates captures
that could retain affected addresses. A warm reserve and hysteresis avoid
repeatedly reclaiming and regrowing ordinary steady-state scratch. Physical
allocation bookkeeping remains charged until the mapping is actually released.

## Interpreting savings

For the local DARK ROAST configuration (131072-token context, target Q8 KV,
microbatch 512), the earlier page-table on/off comparison reduced target CUDA
compute reservation from 705.29 to 232.30 MiB. Target/draft physical KV remained
160/32 MiB, and model GPU weights remained 8291.63 MiB. Whole-device sampled
usage fell by 468 MiB while median warm generation fell about 6.1%.

That comparison predates the new growth/mask/pool switches; it is not a benchmark
of their individual or combined effects. The source also exposes a worst-case
512 MiB full-context K/V-to-F16 temporary and a 128 MiB dense F16 mask for those
dimensions. These allocations have overlapping lifetimes and reservations:
**do not add the estimates or promise the same savings for another model**.
CPU host allocations can grow when a graph's placement changes.

The [small validation record](benchmarks/vram-savings-20261104.json) reports
eleven CPU checks and eight CUDA checks. Compute-only continuations stayed
bitwise identical in twelve dense/hybrid cache combinations per backend. Twelve
CUDA mask fixtures matched dense Flash Attention exactly, and a separate bounded
GEMM fixture returned 34 MiB of actual scratch backing with live inputs preserved.
Repeated allocation denial preserved KV/recurrent state and succeeded after
pressure was removed. Automatic FA fallback and optimizer-mode transitions
restore conventional full reservation before execution.

Analytical VRAM estimates remain conservative rather than subtracting hypothetical
savings. Numerical/resource fixture tests establish their stated correctness
scope; they do not establish trained-model quality or throughput on every GPU.
