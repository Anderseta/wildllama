# Bounded speculative input staging

Patch 062 reduces redundant host work in existing MTP and EAGLE3 paths. It does
not change draft depth, proposal selection, verification, weights, quantization
or requested context. It does not claim performance parity with SGLang.

## EAGLE3 encoder workspace

EAGLE3 already runs its per-token encoder in `n_ubatch` chunks. Concatenation of
the extracted target features now stages one such chunk instead of the entire
`n_batch` prefill. Encoder calls, token ordering, feature values and graph shapes
remain unchanged. The target owns the extracted views while the independent
draft context encodes them.

The feature workspace has at most `min(n_tokens,n_ubatch) * 3 * hidden_width`
floats rather than `n_tokens * 3 * hidden_width`. For an example 512-token batch,
128-token encoder chunk and 4096-wide target, that workspace is 6 MiB instead
of 24 MiB. This arithmetic excludes the target's existing extracted states,
encoded outputs, KV, weights and allocator overhead. No new device allocation
is introduced. Per-sequence indexing/drafting scratch is reused across rounds.

## MTP host fallback and transport budget

A small round can reuse an older, larger prefill device bank. Explicit host
materialization now reads only the mapped active row extent, instead of copying
unused bank capacity. Logical row remapping, captured values and the serialized
carry-state format remain unchanged. MTP sequence-range indexing also becomes
linear in batch tokens rather than scanning all sequences for each token.

`WILDLLAMA_MTP_DEVICE_STATE=1` remains the explicit device-transport opt-in.
`WILDLLAMA_MTP_DEVICE_STATE_MIB` sets the aggregate bank allowance **per inference
context**, from 0 through 32 MiB. Omission preserves the previous 32 MiB allowance.
Zero or malformed standalone environment input disables optional device transport.
This is not the model's total VRAM cap: a target and draft context can each have
an allowance, and existing managed CUDA allocation quotas remain authoritative.
Actual allocated/aligned buffer bytes are charged before publication. Insufficient
optional space retains the verified host transport without reducing context.
Patch 063 also queries the backend's aligned tensor reservation before issuing
an optional allocation, so alignment cannot transiently exceed the transport
allowance and then be rejected after allocation. The managed CUDA quota remains
an independent admission check.

Profiles reload their runner when changing these controls. No profile is silently
changed to enable transport or increase a budget. Native runtime capability
discovery declares both transport controls so older payloads can be detected.

## Validation scope

The standalone `test-speculative-input` contract validates chunked concatenation
against an independently constructed full-prefill reference, workspace reuse,
malformed ranges/overflow, budget boundaries and active-row mappings. Existing
small CPU MTP and speculative-depth tests exercise the real drafting loops.

An additional `test-mtp-device-state --cuda` case checks a real 64-row bank reused
for one active row, a zero/one-MiB allowance, exact hidden values and allocated
byte accounting. GPU execution and trained-model speed/quality comparisons are
deferred when the GPU is occupied; adding that test is not a claim it was run.
