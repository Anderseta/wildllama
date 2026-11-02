# Native response work follows requested fields

WildLlama uses the native server's existing `response_fields` option to request
only completion fields consumed by the Go bridge. Long prompts are no longer
detokenized and echoed back through local IPC when the caller does not consume
them. Native chat and other compatibility response formats also skip this unused
prompt materialization. Generation, tokenization, template processing, sampling,
cache state and memory limits are unchanged.

Direct native `/completion` calls retain the full default response, including
the original prompt. Explicit `prompt` and nested prompt selectors keep the
previous behavior. Older native payloads already support field selection; they
still produce the same bridge responses, although they perform the unused
detokenization internally.

Nested field extraction traverses the source JSON by const reference and copies
only selected output values, preserving flattened paths, missing keys, nulls and
the existing object-only traversal policy. It no longer copies the entire
response once per selected field.

Focused component and Go HTTP tests cover response ownership, field selection,
progress, probabilities, reasoning text, final metrics and speculative counters.
`scripts/gpu-first/test-prompt-response.py` checks default, filtered and explicitly
requested prompts with a tiny CPU-only fixture and reports actual IPC byte
savings. This does not claim a trained-model tokens-per-second improvement.
