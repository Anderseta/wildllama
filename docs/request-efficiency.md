# Request preparation and I/O

These improvements reduce CPU, temporary allocations and abandoned work without
changing model weights, sampling, context limits or numerical kernels.

## Prompt preparation

Consecutive messages with the same role are joined in linear time. Formatting
does not modify the caller's conversation, so rendering the same history again
cannot duplicate its content. Existing separators and first-message metadata
are preserved; individual tool results remain separate.

For text-only rendered chats, preparation reuses the last counted prompt when
it is the selected truncation candidate. It retains at most one candidate per
request. Media tagging and other truncation paths retain their existing flow.

In a small CPU benchmark with 128 text messages, allocations per preparation
fell from approximately 18.1 MB to 0.55 MB for consecutive user messages, and
from 0.894 MB to 0.439 MB for alternating roles. These measurements cover prompt
preparation, not GPU inference or model tokens per second.

## HTTP, downloads and image inputs

- The Go client drains successful responses that have no requested result
  without retaining their entire body. Empty HTTP failures preserve their status
  and authorization errors. Stream framing and its size limit are unchanged.
- Known-size blob downloads validate HTTP status, range and declared length
  before writing. Wrong resumed offsets cannot corrupt a partial file. Fatal
  part failures cancel sibling retry waits promptly. Final size and digest
  checks remain required.
- Image upload reads, validation and publication observe request cancellation.
  Abandoned uploads release their validation slot and remove uncommitted files.
  An upload committed before a lost acknowledgement remains recoverable through
  its existing idempotency key.

Validation uses local HTTP/file fixtures and race tests. No model download or
GPU inference is needed to check these paths.

## Cancellation, metadata and small request budgets

Decision state/image encoding waits for its existing runner slot. Already
canceled batch, tokenization and template requests stop before JSON preparation;
queued admission cancellation aborts the handler chain instead of continuing
into inference.

Cancelable model-show requests can leave a shared metadata lookup promptly.
The shared parse still finishes for other waiters and the cache. Unshared fresh
responses avoid an extra deep copy; cached and shared responses retain separate
ownership. A failed asynchronous lookup cannot publish a cache entry or bring
down the server through panic propagation. Manifest size calculation reads its
layers without changing their shared backing storage.

Uncompressed requests to `/api/embed` and `/v1/embeddings` reserve conservative
output capacity within the configured request memory budget. Small requests can
therefore work with budgets below the former 97 MiB initial estimate. Exact
full-dimension output must fit the already-held reservation before model loading;
there is no second acquisition while decoded input is held. Uncompressed chunked
input is counted without buffering. Encoded requests and the legacy
`/api/embeddings` endpoint retain their original conservative reservations;
compressed wire bytes never become the parsed-input estimate. Body and output
limits remain enforced.

Native capability caching includes library search order and the selected shared
implementation's file identity, size and modification time (`.dll`, `.so`, or
`.dylib`). Replacing the implementation with an unchanged launcher, or adding a
higher-priority implementation, causes a fresh capability query.
Changing payloads are rejected during discovery. These checks occur at runner
startup and do not hash entire DLLs or run in the token-generation loop.
