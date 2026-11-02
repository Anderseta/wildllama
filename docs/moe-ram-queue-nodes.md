# Bounded RAM cache queue nodes

Patch 082 reuses the list nodes that track host weight pages instead of freeing
and allocating them again on every eviction and reload. Nodes are allocated
lazily. The combined active, in-flight and recycled page-node count cannot exceed
the configured weight RAM limit divided by the existing cache page size.

Recycling preserves the exact probation/protected tiers, victim order and stable
resident iterators. It does not change admission, prefetch, SSD read ordering,
weight bytes, physical RAM commitment, pinned staging or GPU memory. Recycled
pointer values are replaced before reuse and never dereferenced.

Failed requests first drain their I/O and release committed physical pages, then
return their pending nodes. This also covers failure while planning a request
before physical pages were committed. The cache retains its existing poisoned
state if physical cleanup fails.

The standalone `test-moe-ram-queue-nodes` uses a counting/failing allocator to
check lazy allocation, the combined node cap, exact order, iterator stability,
exception recovery and allocation plateau after warmup. The small RAM-pool test
also repeats truncated-read recovery with both pooled and legacy physical-page
management. Component timings are not trained-model throughput measurements.
