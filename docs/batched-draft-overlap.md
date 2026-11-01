# Bounded batched and draft response overlap

Patch 064 extends the existing opt-in `WILDLLAMA_DECODE_OVERLAP=1` path.
The original single-slot ordinary-forward transaction is retained. Unsupported
layouts and request types use their ordinary execution and publication path.

## Multiple ordinary sequences

Eligible plain-KV generation can submit one already-accepted token for each
active sequence as one ahead batch. Every output in the completed batch must
have been consumed before another batch is submitted. At most 32 sequences
participate, all must be compatible ordinary text completions, and pending
prefill or incompatible active work prevents admission. This is not a new
concurrency default; existing scheduler restrictions on hybrid families remain.

The context proves CUDA placement for every sampled output. No additional
compute arena, GPU tensors or logical KV capacity is allocated. The normal
backend RAM/VRAM allocation policies remain authoritative. Each sequence keeps
its own entropy and prompt-position checkpoint; cancel/reset drains the shared
batch and rolls back every owned sequence before reuse. Completed tickets are
consumed once without re-evaluating their tokens.

## MTP and other neural drafts

For eligible resident CUDA target/draft pairs, including compatible hybrid
memories, the next draft can start after all current verified tokens have
passed stop/limit checks. Its first successful native submission publishes
the preceding accepted partials while draft GPU work is pending. The callback
is one-shot and scoped to that draft invocation, including exceptions.

This path advances only the existing draft operation. It does not submit an
unverified target token or invent a recurrent rollback snapshot. The extracted
preparation/checkpoint routine is shared with ordinary `pre_decode`; an early
prepared marker prevents duplicate drafting, checkpoint/statistics work and
empty-draft retries on the following iteration.
Any retained/replay draft or zero remaining draft budget blocks early-round
admission, preserving the original controller and replay scheduling.

The first generated partial is immediate. Managed SSD/RAM weight providers,
unknown/ngram history consumers (including mixed fallback chains), unsupported
placement and unknown callbacks retain ordinary publication, so
disk preparation cannot hold a partial until a delayed GPU submission.
Grammar/probability-rich requests, per-token timings, media, indentation/time
controllers and shared-parent requests retain the conservative path.
Early admission additionally requires `prompt_tokens + 2 < context_capacity`.
Drafting does not append target prompt tokens, so the next `pre_decode` cannot
reach its context-shift threshold before consuming the prepared marker.

## Memory and response ownership

The optional batch lazily allocates space for up to 32 entropy checkpoints,
bounded by 512 KiB of host RAM, on its first eligible multi-sequence admission.
Allocation denial retains normal execution. The allocation is reused without
per-token allocation and released after draining on sleep/unload; disabled or
single-slot-only runners do not allocate it. Less than 2 KiB of fixed ownership
metadata is retained independently. This is
runtime metadata outside the existing MoE **weight** RAM limit; that limit does
not cap total process RSS. It introduces no additional VRAM cache or model
weights and does not reduce the selected context.

At most 128 pending partial descriptors reference offsets in text already owned
by live slots, with a 64 KiB aggregate text-span admission bound and less than
16 KiB of descriptor storage. They contain accepted token IDs, request identity
and the original per-token decoded count. No history, vocabulary candidates or
probability arrays are copied into the queue. Response construction uses captured
offsets/counts without temporarily changing live slot statistics or cursors.

Capacity pressure, terminal responses, errors, reset and unsupported/empty drafts
flush partials normally; no accepted text is dropped or truncated. A failed
publisher retains its unsent tail. Existing queue synchronization remains in use.
If draft submission or its publication callback throws, both target and draft
are synchronized before an outer error/reset path can restore or free state.

The standalone CPU helper test uses no model or GPU and covers sequence bounds,
owner isolation, failed-drain retry, partial FIFO/count identity and bounded
publication retry. Actual GPU concurrency, trained-model throughput and latency
still require later measurements; this implementation does not claim universal
coverage, speedup or SGLang performance parity.
