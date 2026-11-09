# Responsive local inference workflows

Non-streaming chat, generate and progress responses stop waiting on their response
channel when the HTTP request is cancelled. A late producer can still finish its
own cleanup, but it no longer retains an abandoned HTTP handler or buffered output
while doing so. Streaming behavior and successful response aggregation are unchanged.

Registry pushes share a blob upload only for the same destination repository.
Preparation, progress and completion are published separately; cancelling one
consumer preserves the others, and the last consumer cancels preparation, network
I/O and retry backoff. A new consumer waits for a closing transfer before retrying.
Fast or registry-mounted uploads still publish final progress. Relative upload
locations are resolved against the registry URL, and parallel parts keep their
authentication retry state separate. These changes do not alter model weights,
inference quality, context or sampling.

Image generation retains a single FIFO GPU execution worker and the existing
exclusive transition lease. Model downloads, local imports and LoRA installation
use a separate bounded I/O worker (one active transfer, sixteen waiting tasks).
They do not acquire image GPU ownership. Installing a pipeline never exposes
unverified partial weights as a usable model. Cancellation, session tombstones
and idempotent request identities apply to both queues; shutdown cancels and
joins both workers.

Job/status/cancel responses copy their public state while locked, then serialize
and write HTTP responses after unlocking. A slow status reader cannot hold the
progress/cancellation mutex while waiting on the network.

## Durable image metadata

Image assets are stored incrementally in `assets.sqlite3` under the image service
root, using SQLite transactions, WAL and full synchronous commits. The existing
SQLite dependency is reused. First publication imports the previous `assets.json`
catalog in the same transaction; that original JSON file remains untouched for
recovery. Committed SQLite rows are then authoritative, including deletions.
Reference ancestry, source ownership and upload fingerprints remain intact.
Deletion tombstones and managed-directory ownership journals stay independent.
No weights, originals or generated PNGs are moved by migration.

On catalog read failure, new submissions/uploads fail closed instead of replacing
the catalog with an empty one. A failed transaction preserves its earlier state.
Session deletion reconciles persisted identities on retry, including files already
removed before a prior metadata commit failed. The legacy JSON is a pre-migration
recovery copy, not a current export: before downgrading to a binary that only reads
JSON, stop the service and export the current SQLite records. Do not remove the
database to force a rollback.

Main history write failures are exposed as `persistence_error`. Small independent
terminal job recovery records preserve results when the main history cannot be
replaced (for example a sharing violation). Restart uses these records only for
jobs still present in the history; they cannot recreate deleted sessions. If both
writes fail, the failure remains explicit rather than claiming durability.

## Disk headroom

Image configuration includes `min_free_disk_mib` (default 1024; zero disables the
guard). Imports account for all selected component bytes before copying. Pipeline
downloads account for the remaining bytes, including resumable partials, and
recheck space during writes. Generation checks output/work volumes before text
unloading, checks again before process launch and monitors free headroom while
the isolated runner executes. Uploads reserve their bounded input bytes.

The guard checks the actual destination filesystem on Windows and Linux. It is
not an OS quota: another process can consume space between observations, and a
native allocation/write may race the polling interval. Known failures report the
disk reason and preserve resumable downloads and existing user files.

## Activity attribution

The runtime activity endpoint serves an immutable scheduler observer registry;
it does not wait behind model construction. Preparation publishes a separate
stage before the runner exists. `activity.request_id` is present only for a
known loading request or an unambiguous active native request. Overlapping native
slots remain unattributed until idle, preventing one chat's shared runner counters
from being labeled as another chat's progress. Request IDs never enter prompts,
sampling, or prompt-cache keys.

These changes preserve model weights, output quality, context and sampling.
Focused CPU fixtures cover stalled HTTP consumers, queue isolation, transactional
migration and rollback, restart recovery, request identity and disk budget math;
they are not evidence of a model tokens-per-second speedup.

## Text-model downloads and uploads

`WILDLLAMA_MIN_FREE_DISK_MIB` sets model-store headroom (1024 MiB by default;
zero disables it). `/api/version` advertises `model_disk_headroom`. Concurrent
GGUF/tensor downloads and blob uploads reserve their known remaining bytes per
actual destination filesystem. Reservations decrease as unique file ranges are
written, so a retry of an already-written range does not consume the budget twice.
Unknown-length uploads acquire space before publishing each input chunk to the
layer writer. A failed guard returns a local disk error without retry backoff;
resumable download data remains available.

Tensor transfer discovery closes an HTTP 200 body without downloading it before
the real request. Copy/hash/progress counters use successful writes, including
partial writes; write errors and short writes cannot be reported as saved bytes.
These are transfer fixes, with no change to inference parameters or model quality.

## Text-model handoff and pending-load priority

Image suspension saves both the resolved mmap choice and its automatic/explicit
origin. Restoration passes that origin through the internal scheduler path; the
next ordinary request can reuse the restored runner without interpreting the
automatic value as an explicit profile change. Explicit mmap settings remain
explicit, and a genuinely different profile still reloads normally.

Inference admission carries chat/background priority into pending model loads.
Requests keep independent prompts, contexts, options and result channels. The
pending selector serves at most four interactive requests while background work
waits; once the oldest background item has waited thirty seconds, it serves at
most one interactive item before returning to background work. FIFO order applies
inside each class, with one bounded exception: one compatible request for the
previous still-resident model may pass a fresh older item. An older item, a queued
profile change for that model, a different context/memory budget, adapter or draft
configuration prevents that shortcut.

The channel and reorder window share the existing maximum pending-request count.
Cancellation releases admission without launching a model; shutdown rejects
queued work explicitly. Already-resident requests retain their fast path. An
in-progress native load is not preempted. The change adds bounded queue metadata,
without new model copies, KV allocations or an increase to RAM/VRAM quotas.
Mock tests establish selection order, fairness, cancellation and reuse; they do
not establish a model throughput or end-to-end latency speedup.

## Fresh profile identity

`GET /api/version` also advertises protocol-1 `features.profile_identity` and a
per-server `instance_id`. A request with `X-Wildllama-Profile-Model` or a `model`
query parameter receives `wildllama.profile_check` with the requested name and
its current named-manifest digest (or a bounded error code). This lookup reads
metadata only: it does not enumerate the model catalogue, read weights or load a
runner. The digest is checked on each probe, so externally replaced manifests
cannot reuse an older profile confirmation. Existing version requests are
unchanged, and clients without this optional feature retain their strict checks.
