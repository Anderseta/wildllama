# Speculative sampling workspace efficiency

WildLlama's rejection sampler verifies drafted tokens against the target distribution. When a draft token is rejected, it constructs the residual distribution `max(0, p - q)` in the target candidate order. Patch `068-sampling-pipeline-efficiency.patch` reduces the CPU work and temporary storage needed by that existing path.

## Dense probability lookup

Previously, large draft supports built a sorted array of row indexes for every rejected round, followed by a binary search for every target candidate. Full-vocabulary distributions now build a direct token-ID-to-first-row table when all token IDs fit in `[0, support_size)`. Construction is linear in support size and each lookup is constant time. Holes return zero and duplicate tokens keep the first row, as before.

The direct table reuses the same `uint32_t` workspace and the existing 262,144-entry ceiling (1 MiB). It never allocates a larger table than the old sorted index. Sparse, signed or out-of-range IDs use the previous sorted-index path. Small supports and allocation failures retain linear lookup. All paths read the original probability values without sorting, normalizing or rewriting them.

## Compact residual entries

Residual entries retain only a token ID and its probability. The unused logit field is removed, reducing each entry from 12 to 8 bytes on supported platforms. The iteration order, floating-point subtraction and accumulation order, uniform random draws and fallback token stay unchanged. Candidate arrays returned to callers are not modified.

These changes apply to any architecture that uses this rejection-sampling path, on CPU or GPU-backed models. They add no GPU tensor, transfer, synchronization or VRAM reservation. They do not alter draft depth, model weights, context length, sampling settings or the target distribution.

## Validation and scope

`test-sampling-probability` compares the previous lookup and residual construction with the optimized path, including 4,096 exact residual draws and RNG state, duplicate and missing tokens, signed and extreme IDs, non-finite probability payloads, allocation failure and workspace reuse. Its vocabulary-sized component benchmark includes index construction and lookup and reports the old sorted-index time alongside the direct table time.

This is a CPU component optimization. Its timings are not model tokens per second, and it is not equivalent to GPU-resident speculative verification or a complete SGLang sampling pipeline. The benefit depends on rejection frequency and the size and density of the draft support. Small truncated supports continue using the existing path.
