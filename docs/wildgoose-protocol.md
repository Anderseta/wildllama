# Wildgoose interoperability

`GET /api/version` retains `version` and the existing `capabilities` array and
adds an optional `wildllama` object. Protocol version 1 advertises boolean
features: `tool_call_ids`, `request_id`, `structured_errors`, `prompt_progress`,
`request_admission`, `embedding_input_limits`, and `multimodal_token_budget`.
Clients must tolerate absent fields, unknown features and external Ollama
servers. This contract does not require a custom prompt or system message.

`X-Request-ID` accepts 1–128 ASCII letters, digits, dashes or underscores. Invalid
or absent values are replaced. The server echoes it and forwards it through
native HTTP calls; it never becomes model input. `X-Wildllama-Queue-Ms` reports
admission wait. `/api/activity` includes a nonblocking `admission` snapshot.
Admission and aggregate-limit errors have `error`, `code`, `retryable` and
`request_id`. Other existing endpoint errors may omit these optional fields.
Clients should preserve a generation ID across retries, honor permanent errors,
and retain ordinary status-based behavior when these fields are absent.

Inference admission covers native and compatibility routes before JSON/multimedia
decoding, including the already-loaded runner path. Interactive work is preferred
with bounded bursts so background work cannot starve. Waiting clients retain no
decoded prompts on this server and cancellation removes their queue entry.
`X-Wildgoose-Priority: interactive|background` can identify retrieval versus
indexing embeddings. This local scheduling hint grants no additional authority.

## Optional server limits

Empty or invalid values use defaults. Changes require restarting the server.
Wildgoose exposes them under its existing advanced server settings.

| Environment variable | Default | Maximum accepted |
| --- | ---: | ---: |
| `WILDLLAMA_MAX_REQUEST_BYTES` | 64 MiB | 1 GiB |
| `WILDLLAMA_MAX_REQUEST_MEMORY_BYTES` | 256 MiB | 8 GiB |
| `WILDLLAMA_MAX_IN_FLIGHT` | 8 | 1,024 |
| `WILDLLAMA_MAX_PENDING_REQUESTS` | 128 | 4,096 |
| `WILDLLAMA_MAX_EMBEDDING_ITEMS` | 4,096 | 65,536 |
| `WILDLLAMA_MAX_EMBEDDING_MEDIA_BYTES` | 48 MiB | 1 GiB |
| `WILDLLAMA_MAX_EMBEDDING_OUTPUT_BYTES` | 32 MiB | 1 GiB |

Byte-valued variables use bytes. The memory limit is an admission reservation,
not an RSS cap for model weights, KV, driver allocations or output encoders.
Embedding requests initially reserve output capacity, then shrink it to their
validated full model dimensions before inference. Requested output truncation
does not conceal full-size temporary vectors. Oversized inputs produce `413`;
a full queue produces a retryable `503` with `Retry-After: 1`. No input is
silently clipped by these guards. Chunked requests receive a bounded input
allowance compatible with the aggregate reservation.

## Stream completion and media

A valid terminal generation frame is followed by at most 250 ms of transport
cleanup before delivering Done/tools and releasing the slot. Normal short
cleanup finishes earlier, preserving callbacks that immediately tokenize.
`[DONE]` without a preceding completion marker remains an incomplete stream.

Native chat budgeting prefers an exact backend media counter when provided.
Older payloads use projector metadata and image headers for Qwen-VL, Gemma,
Pixtral and fixed/tiled projectors, with a conservative unknown-model fallback.
This fallback is an estimate, not an exact multimodal tokenization API. It reads
headers rather than expanding full pixel buffers and supports cancellation.

Tests use local fake HTTP runners, parallel tool/result IDs, queued cancellation,
stalled SSE tails, repeated embedding inputs, profile reconciliation, concurrent
download preparation and capability fallback; no GPU or private models are needed.

## Discarding an interrupted model download

Servers advertising `download_cleanup` in `/api/version.capabilities` accept
`DELETE /api/download` with `{"model":"name:tag"}`. It works offline and removes
only artifacts recorded in that model's local download pin, including GGUF
shards, projectors, and partial-transfer metadata. Installed manifests and other
pending downloads retain shared blobs. Normal model deletion and clearing the
client's operation history are separate actions.

Cleanup returns `removed_files`, `removed_bytes` (logical file sizes, not a
measurement of physical free space), and `retained_blobs`. It refuses concurrent
pulls/imports with HTTP 409, waits briefly for canceled writers to close, and
fails closed on invalid retention metadata. Repeating a successful cleanup is
safe; resuming afterwards redownloads removed files while reusing retained ones.

