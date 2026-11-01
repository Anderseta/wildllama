# Response overlap with sampling, history and constrained output

Patch 060 broadens the existing `WILDLLAMA_DECODE_OVERLAP=1` experiment.
Complete builtin backend chains can now end in either greedy or distribution
sampling and include supported repetition/frequency/presence penalties, DRY,
and an explicit grammar-first backend allow-mask. Ordinary CPU suffixes,
grammar-last rejection, reasoning/time controllers, detailed probabilities,
multiple slots, MTP and unsupported placement retain the original path.

The server accepts the current token and updates its penalty/grammar state
before submitting the next ordinary known-token forward. CPU response
construction does not mutate that sampler state. Only one forward can be
pending; no unaccepted draft token becomes visible to the user.

The existing selected-only guard remains unchanged. Penalty accept handlers
can allocate or throw, so those chains keep their ordinary full backend
outputs. Their candidates are copied into owned host storage before another
forward is launched; response formatting with `n_probs=0` does not inspect the
context's next output row. No new candidate storage is introduced by overlap.

Distribution sampling consumes host RNG while filling graph inputs. A failed
optional launch must therefore restore entropy as well as remove the extra KV
cell. One fixed checkpoint stores only the terminal distribution's two RNG
states and its transactional counters. It does not clone a sampler, penalty
history, vocabulary candidates, grammar or graph buffers. The checkpoint is
bounded at **16 KiB of host RAM per server context**, with no heap allocation
per token and no extra GPU tensors. Greedy checkpoints do not copy RNG state.
It is runtime metadata, outside the existing MoE **weight** RAM allowance;
that allowance has never been a cap on complete process RSS.

The existing model CUDA quota remains authoritative for the normal forward
allocations. The overlap path adds no GPU reservation, context capacity,
speculation width or model weights. It neither bypasses an allocation denial
nor reduces the requested context to recover from one.

Rollback drains backend work before restoring entropy and KV/prompt state.
Checkpoint lifetime/topology identities reject cross-request or replaced
samplers. Nested/custom selectors retain fallback until they supply an
explicit input-transaction contract. Cancellation and reset use the same
single-ticket drain contract as before.

The experiment remains opt-in. Short CPU contract tests cover discarded input
draws, selected token equality, penalties, allow-masks, transactional counters,
and lifetime/topology isolation using zero model weights and a 32-token graph.
They do not establish a trained-model speedup or exercise actual CUDA overlap.
The HTTP/SSE harness now expects the broader admission on supported CUDA
placements; its GPU scenarios and performance comparisons remain deferred.
