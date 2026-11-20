# Warm/cold prefix-cache handoff

Patch `069` improves the existing optional asynchronous serialized-prefix tier.
It does not add a new cache directory, change the file format or introduce a
paged-attention allocator.

## Completed prewrites release their tickets

The writer retains at most two requests and 32 MiB of charged in-flight storage.
Previously a successful request kept its ticket until its prefix was selected
or evicted. Two completed prewrites could therefore prevent every later prefix
from reaching the writer while their RAM copies stayed hot.

Collection now moves the original vectors back to their owning cache entry,
retains the completed immutable disk image and releases the request ticket.
The prefix remains hot: selecting it does not read its disk image. Existing
prompt-cache accounting includes the restored vectors and backing descriptors.
Under pressure, a valid owned backing file lets eviction release vector capacity
without writing the same snapshot again. A missing/truncated backing is replaced
from the valid RAM vectors; page checksums still reject corrupted cold restores.

Cold scoring applies only when the RAM snapshot is absent. A warm entry with a
backing is scored at the measured RAM restore cost, and a failed native restore
may retry or evict that immutable snapshot without another disk write.

The async tier remains explicitly enabled by `WILDLLAMA_PREFIX_DISK_ASYNC=1` and
requires the existing absolute `WILDLLAMA_PREFIX_DISK_CACHE_DIR` and positive disk
quota. Its two-request/32 MiB outstanding-work limits, disk byte/entry quota,
RAM-cache allowance and ownership/teardown rules are unchanged. No GPU buffer is
added. These allowances do not constitute a process-RSS limit.

## Transactional component destinations

Both cold reads and asynchronous handoff reject duplicate/null destination
vectors before publishing any component. An aliased destination previously could
receive two different components in sequence and report success after losing the
first component. Rejection preserves the caller's vectors and the pending job.

## Small CPU validation

The following existing targets cover the new behavior without a model or GPU:

* `test-state-pager`: byte-exact page reads, corruption/truncation, aliases,
  null outputs, disk quota and owned lifetime.
* `test-state-writeback`: original-vector ownership, retained hot backing,
  duplicate handoff rejection, two requests and 32 MiB admission, failures and
  worker teardown.
* `test-server-prefix-cache --state-dir ABSOLUTE_G_DRIVE_TEST_DIRECTORY`:
  four prefixes pass through the two-ticket writer; eight target/draft/spec and
  recurrent-checkpoint warm/cold round trips retain the same file and disk charge;
  hot scoring, missing backing, corrupt cold reads and both quotas are checked.

The tests establish fewer retained writer tickets and no duplicate writes in
those cycles. They do not establish trained-model tokens/s or SGLang parity.

## Incremental local chat checkpoints

Set `WILDLLAMA_CHAT_CHECKPOINT_INCREMENTAL=1` on the WildLlama daemon to enable
incremental saves for eligible local chat sessions. The default is 0. The first
save writes a regular keyframe. Later saves stream the target, recurrent
checkpoints, draft and speculative state to a bounded encoder. Each record
contains either original bytes or an exactly compared span of the keyframe.
A changing header or tensor length can move a reusable span within the file.
The encoder uses about 640 KiB of host buffers and index storage.

Each published delta references one immutable SHA-256-named keyframe for that
session and runtime. The existing disk and per-snapshot limits apply. Restore
validates both files, expands into a temporary ordinary slot within the disk
budget, and calls the same native restore used for full snapshots. A large
delta triggers rotation to a full keyframe on the next eligible save.
Disabling incremental writes still allows existing deltas to be restored.
Incremental saving trades CPU comparisons and keyframe reads for fewer bytes
written. Restoration needs temporary disk space and a complete slot write.
