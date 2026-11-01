# Speculative graph reuse without wider proposals

Patch 066 extends the existing bounded graph arenas to the input contracts
actually used by EAGLE3 and reduces repeated failed admission for short MTP
shapes. It does not pad batches, widen proposals, change the verifier, alter
sampling probabilities or reduce logical context.

## EAGLE3 encoder and decoder

The existing EAGLE3 decoder receives both token IDs and encoded target features.
The ordinary arena filter accepted token-only batches, excluding that decoder.
Its stateless feature encoder was also excluded. The initialized EAGLE3 driver
now grants these two exact capabilities to its context, restoring the previous
capabilities when it is destroyed. There is no repository/model-name list.

Granted token-plus-feature decoder batches must remain non-mixed-row causal
inputs. Feature encoder batches must be tokenless, non-mixed, noncausal, without
a memory context, and use unpooled NextN output. Full topology, inputs, output
counts, sequence identities, placement, adapters and sampler binding checks
remain mandatory on every actual reuse. Encoders do not capture sampler bindings:
feature projection has no sampling transaction even when its decoder has an
attached backend chain.

Every additional arena still requires `WILDLLAMA_GRAPH_VARIANTS_MIB > 0` and
shares that existing per-context budget, including metadata and compute storage.
There are still at most four secondary arenas. The primary reservation remains
available on a miss or denied optional allocation. Managed CUDA quotas remain
independently authoritative. No new default GPU reservation is introduced.

## Short MTP shapes under pressure

Previously, a failed arena attempt for eight or fewer tokens was retried on
every shape transition. The bounded eight-key admission history now remembers
such failures too. It skips 64 observations of that same shape before allowing
another attempt; cache invalidation or bounded aging can also permit a retry.
Transient memory pressure therefore does not permanently disable reuse.

The admission key also distinguishes token/feature/state input kinds and MTP
prediction heads. Denying one input or head cannot suppress another through a
coarse-key collision. Actual tensors are never rounded to these admission keys;
all original `can_reuse` checks still apply.

## Response work after neural submission

An internal draft overload accepts a single-shot callback immediately after a
successful neural submission and before its sampling/output synchronization.
It permits the server to format already-owned responses while the draft runs.
The callback does not run for zero-width drafts, lookup-only implementations or
failed submissions. Its binding is restored on every exit, including exceptions.
It must not mutate the drafter or either inference context.
Chained MTP prediction heads also restore their normal head offset through an
exit guard, including when the callback throws before sampling.

The ordinary multi-sequence overlap query proves that every sampled output is
one token on the same CUDA backend. The draft-response query additionally
supports resident hybrid graphs without requiring a sampled output. Both reject
unsupported placement, weight streaming/providers, explicit tensor overrides and
CPU execution. Draft admission rejects encoder/media input and requires aligned
coordinates for text M-RoPE. These capability queries allocate no GPU buffers.

## Validation scope

`test-speculative-graphs` uses only small in-memory CPU fixtures. It alternates
prefill, neural drafting, target verification and accepted/rejected boundaries,
compares logits/proposals/carry bytes with graph retention disabled, checks real
retained encoder/decoder/verification hits and verifies metadata plus compute
bytes against each selected budget. Attached draft top-k chains and callback
exception/zero-width cases are included. `test-graph-variants` also tests the
short-shape cooldown, retry and head/input isolation.

CPU fixture reuse is not a trained-model tokens/s result. GPU performance and
language quality measurements require a separate comparison when the GPU is
available; no superiority over SGLang is claimed.
