# WildLlama runtime performance

This private fork keeps model files, quantization and the requested context unchanged.
Native changes live in the numbered `llama/compat/*.patch` stack against the
revision pinned in `LLAMA_CPP_VERSION`, including the original runtime-performance
patch and the newer memory/lifecycle changes described below. These are fork changes,
not an upstream PR.

## Automatic improvements

- Prompt-cache restoration compares estimated saved prefill time with observed
  restoration cost once both measurements exist. Cold caches retain the previous
  heuristic. Matching remains exact, including recurrent checkpoints.
- Explicitly noncausal GGUF pooling models can batch independent text embeddings.
  Tokenization is batched, work is bounded by physical tokens and sequence count,
  duplicate texts are computed once, and results retain input order. Media and
  incompatible runners retain their existing path. A runner 413 retries only the
  affected group through the existing truncation logic. Cancellation and errors
  propagate. `OLLAMA_EMBEDDING_MAX_SEQUENCES` selects 1–32 slots (default 8 for
  eligible models); 1 restores the sequential runner behavior.
- Resident-model eviction considers measured reload cost, reuse, recency and bytes
  released among models with the same keep-alive priority. Idle models are still
  preferred. Missing observations retain the previous ordering.
- A bounded, process-local placement cache records fully GPU-resident dense loads
  with allocator headroom. Signatures include model timestamp, devices, drivers,
  context, runner options and relevant environment. Partial placements, MoE,
  projectors, auxiliary drafts and uncertain observations use existing estimates.
- Chat truncation keeps tool results with their assistant calls, preserves system
  messages and prepares cut boundaries in one pass. Rendered token counts and
  native recurrent/SWA checkpoint checks remain authoritative. No approximation of
  recurrent state or new media-token heuristic is introduced.

## Per-model runner settings

`GET /api/version` advertises `runner_env`, `num_ubatch` and `embedding_batch`.
`options.runner_env` is a validated object of native runner settings. It can also
be persisted through `/api/create` in `parameters.runner_env`. Values affect the
child runner, without changing the server process environment. Paths, credentials,
model selection and arbitrary environment variables are rejected.

For example, a reference model can retain an independent cache and physical batch:

```json
{
  "model": "example-profile",
  "from": "existing-model",
  "parameters": {
    "num_ubatch": 128,
    "runner_env": {
      "LLAMA_ARG_CACHE_RAM": "1024",
      "LLAMA_ARG_FIT_TARGET": "4096",
      "OLLAMA_KV_CACHE_TYPE": "q8_0"
    }
  },
  "stream": false
}
```

Allowed families: load/lazy/fit, GPU and CPU-MoE placement, MoE GPU cache and RAM
weights budget, prompt RAM cache, logical/physical batches, KV cache types, flash
attention, parallel sequences, tensor placement and warmup. Empty values clear an
inherited runner override. Explicit API options and command-line settings retain
their existing precedence. The RAM budget still covers MoE weights, not total
process memory or dense offload.

### Adaptive prompt RAM admission

Native runners advertising `WILDLLAMA_PREFIX_RAM_ADAPTIVE` use an adaptive ceiling
for optional serialized prompt snapshots when the configured prompt RAM cache is
positive. The Go launcher enables the capability only on a compatible executable;
older payloads retain their original behavior and do not inherit the switch.

The user cache setting remains the upper bound. Admission additionally reserves
5% of physical RAM, clamped to 512–2048 MiB, using OS available-memory information.
Under pressure the existing eviction/spill path releases old optional snapshots;
an oversized snapshot is skipped before serialization. Pending allocations are
accounted once, and the active request's tokens and live KV/recurrent state are
preserved. A smaller cache can require more prompt processing on a later revisit.

Unknown OS memory data retains the configured ceiling. The optional native
shared-cell retention layout also retains its existing policy: its logical GPU
charge must not be mistaken for serialized host RAM. This is cache admission,
not a hard process RSS/commit limit and not a reduction of model context or KV
precision. It creates no new cache directories and keeps the configured disk
tier optional.

`num_ubatch` separates the physical microbatch from `num_batch`; zero/omitted keeps
the previous behavior. Smaller physical batches can save scratch space but may
reduce prefill throughput. Whole-prompt pooling/decision models reject a microbatch
smaller than their required physical batch rather than silently splitting it.

Wildgoose detects this capability. Applying load-affecting settings unloads only
the affected model. Sampling-only changes keep its weights resident. Truly global
settings such as maximum loaded models and keep-alive can still require a daemon
restart; official Ollama retains the legacy profile behavior.

## On-demand visual projectors

The launcher enables `WILDLLAMA_MMPROJ_ON_DEMAND=1` only when a runner advertises
it through `--wildllama-runtime-features`. A vision-only MTMD context initially
keeps tensor metadata and preprocessing rules without encoder weights or device
buffers. Vision remains available to the model. The encoder loads when an image
needs encoding and releases its weights and compute buffers once no slot has a
pending prompt to process; generation, cached image embeddings and text KV remain
valid. Re-encoding images in a later request loads it again. Prefix reuse that
already contains the image embeddings needs no reload.

`runtime_memory.projector_state` and `projector_progress` report the observed
lifecycle. Released projector buffers are removed from placement telemetry. Each
cold encoder load has a latency cost and still needs enough memory alongside the
text model and its current KV. Allocation failure returns a request error without
resetting the text model. In-flight encoding retains ownership until it finishes,
including cancellation. Audio and mixed audio/vision projectors retain their
conventional lifecycle; older payloads retain their original loading behavior.

## Live KV allocation measurements

Compatible native contexts publish `runtime_memory.kv_committed` with physical
`gpu_bytes`, `host_bytes`, logical `capacity_bytes`, a timestamp and source.
Measurements run on the context owner thread, are throttled during work, and are
refreshed at idle after trimming. They follow KV growth and release rather than
reporting only its startup allocation. These counters exclude model weights,
recurrent state and compute buffers. An unsupported layout or older payload leaves
the measurement absent; its load-time KV counters remain a separate snapshot and
must not be treated as current allocation.

## Optional recurrent prompt checkpoints

With `WILDLLAMA_PREFIX_RAM_ADAPTIVE=1`, per-slot optional rewind snapshots also
use the available-RAM reserve described above, independently of whether the global
prompt cache is enabled. Before allocating large target/draft copies, admission
can release older optional snapshots or skip the new snapshot. Existing allocated
vector capacity is counted; pending bytes are charged once. Allocation failure
leaves the current request running, and only complete snapshots enter the list.

Coalescing retains spaced interval anchors and the two newest checkpoints. The
configured count limit and spacing setting remain unchanged. It removes nearby
intermediate snapshots even before the list reaches its count limit. Short chats
therefore avoid accumulating a full recurrent-state copy at every nearby boundary.
An older edit can require replay from an earlier anchor or the full prompt. The
prompt tokens, live KV/recurrent state and mandatory speculative rollback state
are preserved; context and sampling are unchanged. This remains optional-cache
admission, not a hard process RAM limit. Unknown OS memory information preserves
the existing admission fallback. Explicit restoration of checkpoint appendices
from manually saved slot files retains its existing parser and allocation policy.

## Optional experiments

Additional MTP verifier, proposal-head and compact recurrent-state controls are
documented in [MTP decode efficiency](mtp-decode-efficiency.md), including their
memory costs, compatibility checks and correctness fallbacks.

These are exposed in Wildgoose profiles and remain disabled/omitted by default:

- `backend_sampling: true` tries native backend token selection. Grammar, reasoning
  budgets and incompatible samplers retain the CPU path. Existing filters and
  penalties are preserved. Shader startup can increase the first-token delay.
- `speculative_type`: `auto`, `none`, `ngram-simple`, `ngram-mod`, `ngram-cache`.
  N-grams require no extra model; they still use target-model verification.
- `draft_num_predict` bounds proposed tokens. `speculative_adaptive: true` measures
  ordinary generation and total proposal/verification cost, cautiously probes,
  reduces ineffective drafts and retries after a cooldown. It cannot make every
  prompt faster and does not enable missing MTP weights.
  Probes use the initialized technique's native span: tiny drafts can lose solely
  to fixed recurrent-checkpoint costs and incorrectly hide profitable long drafts.

Greedy-output equality, embedding cosine/ranking comparisons and grammar checks
are useful regressions, not proof of unchanged quality on every possible input.
Hardware and model measurements belong in a separate benchmark report; cold load,
warm generation, prompt processing and cache restoration must be reported separately.
