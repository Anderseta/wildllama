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
