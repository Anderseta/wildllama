# Output and snapshot pipeline efficiency

The [small validation record](benchmarks/output-pipeline-20261102.json) covers
patches 075–078 on Windows with a shared RTX 5060 Ti. No trained models were
loaded or downloaded. Twelve focused CPU cases, three bounded CUDA modes and
six before/after HTTP cases per payload passed. Wildgoose's 79 focused UI cases,
TypeScript checks and package-source guard also passed.

## Automatic output work

Eligible [host output copies](batched-host-outputs.md) use the existing arrays
and synchronization boundary. The real tiny MTP test recorded six coalesced
copy runs, with unchanged public tokens, candidate counts, probabilities and
logits. Pinned and ordinary pageable destinations preserve payload and padding.
Unsupported rows keep their previous copy commands; single-row output vectors
skip extension lookup.

Indexed output ordering reuses the existing token-to-output map and emits the
exact same deferred swaps as selection sort. Small or sparse maps retain the
original path. The 480 permutation cases include malformed-map rejection before
mutation; no extra workspace is allocated.

| Component control | Previous | New |
| --- | ---: | ---: |
| Reverse 512-output order, 256 repetitions, four ABBA cycles | 72.14 ms median | 0.75 ms median |
| 512 four-byte output rows, 128 repetitions, two ABBA cycles | 2,385.61 ms median | 9.46 ms median |
| Copy submissions per eligible 512-row run | 512 | 1 |
| Managed CUDA peak in output-copy control | 4 MiB | 4 MiB |

These timings isolate metadata work and driver submissions, rather than neural
inference. They do not predict a proportional tokens-per-second improvement;
ordinary single-token output and unsupported layouts can see no gain. Shared
GPU activity and Windows driver scheduling affect the observations.

## Optional cooperation and SSD reuse

The [cooperative verifier](bounded-device-verification.md) fixture retained its
8,192-token logical context and 128 MiB managed allowance. It observed one actual
target physical-growth event and nine child preflight reclamations. Target and
draft serialized state and token sequences matched the reference; stable decode
reused scratch, and child restore released it before mandatory work. Peak
tracked allocation was 131,484,032 bytes under 134,217,728 bytes. Unknown/shared
layouts retain fallback. This resource protocol remains opt-in.

[Shared cold-prefix chunks](hierarchical-prefix-pages.md) restored complete
components exactly, including shifted/grown serialized-like data, concurrent
writers, retained origins, corruption and failed optional writes. A retained
branching-prefix fixture reused 1,769,490 bytes in 11 chunks, with 530,040 bytes
charged against its 4 MiB metadata allowance. Every retained origin keeps its
full physical disk charge. Existing obsolete-prefix pruning is preserved.

The structured-padding test exposed a real lookup weakness: different chunk
lengths had different full checksums but identical low bits. Bucket selection
now mixes all checksum bits; full checksums and byte-by-byte reuse proofs are
unchanged. The fixed index, metadata allowance and 64-file limit are unchanged.

SSD sharing remains disabled by default and needs an explicitly configured
directory and disk quota. It reduces duplicated snapshot writes where matching
sources remain owned. It still materializes complete components in host memory
and does not add general paged attention or SSD-backed weight-inference speed.
