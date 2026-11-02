# Batched uploads for the normal MoE expert cache

The normal GPU expert cache now submits the selected gate/up/down weight ranges
through the existing RAM-cache batch-copy API. Previously each contiguous expert
run made a separate synchronous managed copy, even after a combined prefetch.

The normal-cache path keeps the same source offsets, destination slots, original
quantized weight bytes, cache replacement policy, router decisions, and configured
RAM/VRAM limits. It adds no device bank, pinned upload ring, or speculative reads.
All selected managed pages remain pinned until the admitted upload batch has
finished or handed off to the compute stream through its existing event protocol.

For ordinary nonpooled host weights, the existing pinned staging implementation
can finish a batch with one backend fence instead of a fence for each expert run.
Only devices advertising asynchronous transfers and host staging buffers use
this optimization. Eligibility is cached once when the device cache is created;
backends without those capabilities use the original prefetch/per-run path
directly, without constructing a batch descriptor list.
The pooled upload queue retains its existing overlap behavior where supported;
this change does not introduce that queue into conventional nonpooled models.

If the selected page union exceeds the existing RAM budget or contains an
unmanaged source, batch admission performs no uploads. The normal cache then
uses its established prefetch and per-run managed/direct-copy fallback. Failed
reads or transfers propagate through the existing slot-map rollback guard, so
new cache mappings are not published against incomplete weight uploads.

## Small regression tests

`test-moe-normal-upload DIRECTORY [--gpu]` compares the production normal-upload
helper to the original per-run transfer order using synthetic raw weights. It
checks exact expert bytes and untouched bank slots, cold read counts, warm reuse,
managed RAM limits, backend fences, over-quota and mixed-source admission denial,
validation failure in a later plane, and truncated-read failure. Its timing is an
operator measurement, not a model token-throughput claim. A capability-disabled
case requires exactly the legacy bytes, fences, read calls and managed residency.
The CPU helper is exercised for correctness only; production uses the original
path for CPU-like or otherwise unsupported transfer capabilities.

`test-moe-normal-upload DIRECTORY --gpu --model tiny.gguf` additionally decodes a
small untrained MoE fixture through the public context API. It requires actual
normal GPU-cache misses and compares every logit with the resident-weight normal
cache. Models larger than 32 MiB are rejected by this test. The model sessions run
sequentially under a 128 MiB managed GPU test quota; the test harness must use the
same process-wide quota. Production model settings are untouched.
