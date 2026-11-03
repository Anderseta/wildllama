# Conversation image cleanup

`GET /api/images/capabilities` advertises `session_deletion: true`.
`DELETE /api/images/sessions/:session_id` removes managed input/output files,
job directories and catalog entries owned by that conversation. A successful
response contains `ok: true`, `session_id`, `removed_jobs`, `removed_assets` and
`retained_shared_assets`. Blank IDs are rejected; standalone image-tab jobs
remain independent.

Deletion persists a tombstone before cancelling work, then waits for the worker
and any upload to release their files. HTTP 409 with `retryable: true` means
cancellation is still completing; retry the same deletion. Late uploads, job
submissions and reuse of deleted request IDs are rejected across restart.
Images referenced by another conversation remain until its final reference is
deleted. Models, LoRAs, original external files and conditioning caches are
preserved.

Output ownership is persisted separately from bounded job history, including
failed jobs' partial directories. Restart or history pruning does not discard
that ownership. An unreadable deletion/ownership journal blocks new work rather
than silently forgetting deleted conversations.

For old histories whose native job records were already pruned, an optional
JSON body can supply `legacy_asset_ids` (up to 4096 per call). Only catalog assets
without another known owner can be associated this way; other conversations'
references still protect them. Repeating deletion is safe. Client paths are
never accepted, and deletion stays within recorded managed output roots without
traversing directory links.
