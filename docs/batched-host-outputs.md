# Batched host sampling outputs

Patch 077 adds a bounded CUDA extension for copying compatible sampling rows to
their existing host output arrays. It targets repeated driver submissions, not
model arithmetic, quantization or the sampling distribution.

The ordinary path can issue one device-to-host copy per row for selected token
IDs, probabilities, logits and candidate IDs. Consecutive small rows can use one
contiguous transfer or one pitched two-dimensional transfer when their original
allocations and destination layout prove that the byte payload is identical.

## Admission and memory

The planner examines at most 1,024 original rows per run. Admission requires:

* one CUDA producer backend and one ordinary default CUDA buffer owner;
* contiguous F32/I32 vector rows with equal widths, at most 64 KiB each;
* increasing, uniform and nonoverlapping source pitch, with every row inside its
  actual allocation;
* aligned host strides, sufficient destination capacity and no source/host alias;
* pitches supported by the current physical CUDA device.

Null rows, different widths or owners and reordered layouts end a run. A run
shorter than two rows retains the original copy path. Sparse VMM, host, custom,
split and other backend buffers remain excluded. No oversized synthetic tensor
is used to bypass the bounds of the original row tensors.

The extension allocates no device or host staging buffer. It keeps only a small
thread-local cache of an immutable device pitch limit. The existing output
buffer, row strides, per-row candidate counts and synchronization boundary stay
unchanged. Only the submitted copy commands differ; padding between rows is not
transferred. GPU model and KV budgets retain their existing scopes.

The registry extension `ggml_backend_cuda_output_api` is versioned. A return of
zero submits no copy and permits ordinary fallback. Device errors after valid
admission retain the original CUDA error behavior. The row descriptors are
consumed before return; the original source and destination storage must remain
alive until the caller's existing output fence, as before.

## Small validation

`test-host-output-copy` checks pure CPU planning boundaries. Its `--cuda` mode
uses no model, a 16 MiB managed CUDA allowance, original tensor/scheduler objects
and pinned host destinations. It compares all bytes, including unchanged host
padding, for scalar and compact rows, and verifies rejected layouts before
submission. A separate tiny ordinary `std::vector` destination checks exact
payload and untouched padding for the pageable host fallback. The
[CUDA Runtime memory API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__MEMORY.html)
allows wholly pageable transfers; a transfer must not span mixed registered and
unregistered host regions. Existing output arrays use one owned allocation. No
new host registration or pinning policy is introduced.

A warmed ABBA component benchmark compares 512 individual row copies
with one batched copy while checking unchanged managed GPU memory.

The component benchmark does not establish trained-model tokens per second.
Independent workloads sharing the GPU can change timings. Default-path numerical
and protocol comparisons must also cover stochastic and greedy sampling, sparse
rows and speculative output consumers after the context integration is built.

For correctness-only integration diagnostics, `GGML_CUDA_OUTPUT_TRACE=1` emits
one terse stderr record after each successfully queued batched copy, including
the row count, byte width and source/host pitches. Rejected layouts produce no
record. The toggle is read once; leave it unset for performance measurements.
