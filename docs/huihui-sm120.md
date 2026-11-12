# Huihui IQ3_S inference on SM120

The WildLlama launcher enables IQ3_S sign expansion on a single SM120 CUDA device and offers coefficient reuse to MTP on the same device. The native kernels check the measured matrix shapes and compact recurrent layout before dispatch. The profile can override either default with `0`. GPU block verification and the packed proposal head remain opt-in.

| Runner setting | Scope | Additional device storage |
| --- | --- | --- |
| `WILDLLAMA_CUDA_IQ3_FAST_SIGN=1` | Exact sign expansion in the one-token IQ3_S kernel for five measured SM120 matrix shapes | None |
| `WILDLLAMA_MTP_GPU_VERIFY=1` | CUDA acceptance and correction for supported probabilistic MTP blocks | Existing `WILDLLAMA_VERIFY_DEVICE_MIB` budget must be positive |
| `WILDLLAMA_MTP_PACKED_HEAD=1` | Contiguous Q4_K proposal weights when an indexed proposal vocabulary is configured | Up to 128 MiB, including row indices and updates |
| `WILDLLAMA_GDN_REUSE_REPLAY=1` | Reuses decay and delta coefficients when replaying extended compact GDN state | None |

GPU verification reads the target probabilities and candidate IDs on the device. The host supplies bounded proposal probabilities and random draws and receives the accepted prefix length and correction token. It commits the draws only after a successful device decision. Grammar, finite reasoning budgets and active forced reasoning stops use the host path. The server still owns the speculative transaction, state commit and request scheduling. The existing raw-logit fallback remains available when a caller requires it. Floating-point probability preparation can produce a different trajectory for the same seed.

The packed head keeps exact Q4_K rows, uploads only changes and orders the gather on the proposal stream. It skips the comparison when the shortlist revision is unchanged. Its device allocation is admitted through the backend budget. It has a separate bounded pinned upload buffer. Each proposal still uses its actual distribution for target verification. The separate contextual vocabulary can change acceptance independently of the packing layout.

Compact GDN replay keeps the original FP32 arithmetic order for each state element. It moves the loop over updates outside the per-thread row loop so one thread can reuse the shared coefficients. It applies to the existing sequential compact rollback layout for three or four proposals.

The launcher keeps the configured proposal count. A profile with two proposals continues to use the two-proposal rollback path. When running the native executable directly for a benchmark, set the two launcher defaults explicitly to reproduce normal server dispatch.

## Component measurements

The RTX 5060 Ti measurements use CUDA 12.8. They compare real Huihui quantized matrix slices or small deterministic fixtures. They do not measure total tokens per second.

- IQ3_S sign expansion improved five one-token projection shapes by 5.4–7.2%, with bitwise equality. The same-size reversible packing prototype passed 189 comparisons across three variants. Packing with column reuse improved four one-token shapes by 10.8–15.1% but regressed most three-token shapes. The prototype remains in the fixture because the other GGUF consumers require explicit layout support.
- The compact GDN replay fixture passed all 75 comparisons across five starting rows, five continuation lengths and three activation patterns. In five-token rounds, reconstruction of an intermediate row took 12.2–13.5 µs with coefficient reuse versus 19.8–26.1 µs before. Starting at the newest state took about 11.3 µs in both cases. Neither variant allocates a second state buffer.
- A Release probe of the packed Q4_K proposal head used the production gather on the projection stream and preserved all tested output bits. For 4,096 fixed candidates, the indexed projection took 78.0 µs and the packed projection took 52.5 µs. At 16,384 candidates those times were 134.9 and 179.3 µs. Cache storage was 11.30 and 45.19 MiB respectively. Refreshing every eight proposals took 129.2 µs for the packed 4,096-row case, versus 60.7 µs indexed.

The IQ3 layout experiment draws on the [SYCL IQ3 reordering work](https://github.com/ggml-org/llama.cpp/pull/29107). The packed proposal experiment follows the layout idea in [NanoSpec](https://arxiv.org/html/2605.26444v2). The recurrent path still verifies a sequence of proposals and reconstructs the chosen state, without a tree scheduler or changes to the model's state precision.
