# Shared cold-prefix chunks

`WILDLLAMA_PREFIX_DISK_PAGE_CACHE_MIB` optionally enables block sharing in the
existing serialized-prefix SSD tier. Zero/unset keeps the original format and
behavior. Values 1 through 64 select an additional CPU metadata allowance in
MiB. The existing absolute cache directory and disk quota are still required;
the setting does not choose a directory or reserve GPU memory.

## Exact sharing across snapshots

Snapshots use content-defined chunks between 192 and 256 KiB. A bounded rolling
hash window chooses stable boundaries rather than assuming that serialized
headers or KV rows begin at the same offset in every snapshot. This can reuse
unchanged regions after a header shifts or a prompt grows. A bounded direct
index locates candidates, and a complete byte comparison proves every reuse.
Hashes alone never authorize sharing.

Each new snapshot writes one manifest and only its new chunks. References point
directly to an immutable owned source file and byte range, not to another
snapshot's descriptors. Old snapshot metadata can therefore disappear without
forming an ancestor lookup chain. The format remains process/session-local;
cross-restart persistence is not added.

Sharing requires the source snapshot or one of its referenced chunks to remain
owned. Existing prompt-cache pruning is unchanged: a linear extension can remove
an obsolete predecessor before the new snapshot is spilled, eliminating that
reuse opportunity. Retained branches and independent live snapshots can share
matching regions; the weak candidate index does not keep obsolete sources alive
just to improve its hit rate.

Restore validates the manifest, component lengths, source identities/ranges and
every chunk checksum. All components are staged before the original atomic
publication. Missing, truncated or corrupt origins fail the cache restore and
retain the normal prompt-recomputation path. Warm snapshots check availability
of their referenced files before discarding valid RAM; an unavailable backing
is rewritten from those valid components.

## Bounded resources

The pool has an 8,192-slot weak candidate index with a conservative 512 KiB
charge. It reserves descriptor capacity, component/entry metadata, path/file
ownership and bounded working storage before admitting shared storage. A store
uses one 256 KiB comparison buffer plus a fixed stream/allocation envelope;
concurrent stores share the selected metadata allowance. Failed admission or
optional I/O retains the original complete-snapshot path when that path fits.

Shared entry metadata also contributes to the existing prompt-cache RAM charge.
The asynchronous writer reserves its descriptor upper bound before moving input
vectors and keeps its existing two-request/32 MiB limit. These accounting scopes
do not claim a total process-RSS bound; serialized payloads and model memory keep
their existing separate allowances.

Physical files retain their complete disk charge while any snapshot references
one of their chunks. A small child can consequently keep a large original file
alive. The existing 64-file limit counts those origins and active writes, and
failed removal retains its physical charge. No background file compaction or
unbounded page-file collection is introduced.

## Scope and validation

The model-free targets add modes for the new path:

* `test-state-pager ABSOLUTE_DIRECTORY --pages`: shifted/grown serialized-like
  components, exact restores, retained origins, the 64-file limit, 96 generations
  without descriptor chains, four concurrent stores, corruption, metadata denial
  and Windows partial-write fallback.
* `test-state-writeback ABSOLUTE_DIRECTORY --pages`: metadata admission before
  moving original vectors, exact handoff/restore and both resource budgets.
* `test-server-prefix-cache --page-state-dir ABSOLUTE_DIRECTORY`: cold metadata
  in the RAM accounting, full component continuity and warm rewrite after an
  origin becomes unavailable.

This saves duplicate SSD payload writes and disk space for matching regions.
It does not stream KV pages directly into a running model, avoid the existing
whole-component host materialization, add general paged attention, or establish
a trained-model throughput improvement. Weak-index collisions and changing
content can reduce reuse without changing snapshot bytes or inference behavior.
