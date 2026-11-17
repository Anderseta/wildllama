# Few-row MMQ and graph-arena host budgets

Compatibility patch 113 backports the [upstream few-row CUDA MMQ kernel](https://github.com/ggml-org/llama.cpp/commit/0fb8f1577) to the pinned native backend. It is **opt-in**, with `WILDLLAMA_CUDA_FEW_ROW=1`; unset/0 retains the existing dispatch. Selection is immutable for the process so captured CUDA graphs remain valid. Unsupported types, alignments, shapes and devices retain the original path. HIP/MUSA do not use it.

The kernel targets small matrix batches, not one-token GEMV. The inherited dispatch thresholds vary by quantization and device. In particular, IQ3_S normally starts at five columns, while Coder390 with MTP depth two usually verifies fewer columns. A faster component therefore need not improve this model's generation speed.

Patch 114 accounts for scheduler host reservations before admitting optional retained graph arenas. This includes tensor-copy contexts, IDs, hashes, copy tables and allocator bookkeeping. Overflow rejects admission. Quotas include host reservations plus graph metadata and compute buffers; insufficient space falls back to the primary scheduler. This is a conservative reservation, not an RSS or CUDA-driver-memory measurement. Ordinary CUDA graphs are unaffected. Extra graph arenas remain disabled by default.

## Local validation

Windows, RTX 5060 Ti 16 GiB, driver 621.24, CUDA 13.5.37, SM 120a; Ryzen 5 2600. Native base: `9ff0d7e90f0faaa7c16017e6dcf9cf91c50a100e` plus the compatibility stack through 114.

`scripts/check-cuda-few-row.py` uses 96 synthetic MUL_MAT cases per mode: eight quantizations, six batch widths, and two aligned/tail shapes. Both original and opt-in dispatch passed all 96 numerical cases against the test backend's reference/tolerance. This is numerical validation, not bitwise identity or a model-quality benchmark.

Measured old/new operation-time ratios (larger is faster; mean of the two shapes):

| Quantization | 3 columns | 8 columns |
| --- | ---: | ---: |
| Q3_K | 1.28 | 1.47 |
| Q4_K | 0.99 | 1.42 |
| Q5_K | 1.27 | 1.52 |
| Q6_K | 0.96 | 1.16 |
| Q8_0 | 0.97 | 1.60 |
| IQ3_S | 0.99 | 1.41 |
| IQ4_NL | 1.03 | 2.12 |
| IQ4_XS | 1.00 | 1.89 |

These synthetic ratios are not end-to-end tokens/s. Whole-model trials use Coder390 IQ3_S3, 40,960 context, 8,192 initial KV, Q8_0 KV, fixed MTP depth two, and the existing sampling settings. The vision projector is not loaded. Two synthetic requests (code and SQLite explanation) each have a warm-up and a 384-token timed generation, with dispatch off/on/off. Results vary with sampled text and draft acceptance; they do not establish a consistent benefit, so the kernel stays opt-in. No personal profile is modified.

| Timed generation | Off, first pass | On | Off, repeat |
| --- | ---: | ---: | ---: |
| Code, tokens/s | 38.50 | 35.05 | 46.30 |
| SQLite explanation, tokens/s | 35.37 | 37.60 | 37.79 |

The repeated baseline also varied substantially. This short trial cannot isolate a reliable whole-model speedup. Native telemetry confirmed GPU-resident KV (320 MiB at initial capacity, zero host KV) and functioning MTP block verification. The test limits generated tokens only for measurement; the user's output limit is unchanged.

Graph-budget checks pass on small local CPU and CUDA fixtures: exact logits, insufficient-quota fallback, invalidation and arena reuse. CPU sampling is checked with a fixed seed; CUDA uses a greedy backend oracle because independent baseline GPU contexts themselves did not reproduce stochastic samples exactly. The test includes a 64 MiB quota that must reject the optional arena now that scheduler host memory is counted. A sufficient 512 MiB quota exercises reuse.

Reproduce with a prepared native tree and `scripts/gpu-first/build-windows.ps1`, targeting `llama-server`, `test-backend-ops` and `test-graph-variants`. Run the numerical matrix only when the GPU is available. The graph test accepts a local fixture no larger than 64 MiB and never downloads a model. Keep benchmark artifacts outside the repository's tracked files.
