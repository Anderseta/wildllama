# Bounded decoding experiments

These controls are independent, disabled by default, and accepted in a model's
`runner_env`. Wildgoose discovers support from the selected native executable,
stores the choices in each model profile, and applies them when reloading it.
Use `"0"` or `"1"`; changing a process environment after loading a backend does
not change its cached settings.

| Control | Scope | Additional storage |
| --- | --- | --- |
| `WILDLLAMA_NGRAM_FEEDBACK` | Online proposals in explicit `ngram-simple` mode, any supported target architecture | 74,064 bytes of CPU state per sequence in the tested build; up to eight draft tokens |
| `WILDLLAMA_CUDA_TOKEN_REUSE` | Q4_K dense projections and compatible Q4_K/IQ3_S routed projections | Registers; no persistent weight copy or device buffer |
| `WILDLLAMA_CUDA_L2_REUSE` | Quantized matrix-vector activation accesses on supported CUDA devices | At most 1 MiB and 1/16 of existing L2 capacity; no additional VRAM allocation |

The online proposer records only target-confirmed history. Full suffix checks
prevent hash collisions from becoming token matches. Rejections reduce the
priority of a continuation; new confirmed occurrences can rehabilitate it.
Tables are bounded and reset between requests and after history truncation.
Every proposed token still goes through the existing target sampler and rollback
path. This option does not replace a selected MTP drafter, change its sampling,
load another model, or reduce the target vocabulary.

The token-reuse kernels retain the original packed weights and per-output
arithmetic/reduction order. Routed reuse requires equal expert IDs at the same
router rank for adjacent tokens. Different experts retain independent outputs;
there is no routing approximation. Fused projections and other quantization
formats use their existing implementations. Reuse can lose performance through
register pressure or insufficient matching routes, even without allocating more
VRAM. Synthetic kernel gains are not whole-model throughput gains.

The L2 policy describes only a known contiguous quantized activation row. It is
attached to kernel launches, including captured CUDA graphs, rather than relying
on a mutable stream setting. Existing programmatic dependency launch attributes
are preserved. Unsupported backends, devices and unavailable cache reservations
retain normal accesses. L2 prioritization competes with other data in the cache;
it is a performance experiment, not a memory-capacity optimization.

The changes do not convert GGUF formats, prune experts, drop context or change
model quantization. They are independently implemented adaptations of general
online speculation and reuse ideas; they do not import an offline corpus or the
custom weight formats used by llama-modchip.

Validation includes `test-ngram-feedback`, `test-cuda-token-reuse` (pass any
argument for paired CUDA-event timings), and filtered `test-backend-ops` runs.
The CUDA test compares output bits and guard regions for dense and routed
projections, odd row/token counts, matching/different experts and strided inputs.
End-to-end timing must separately check physical RAM, Windows commit headroom,
page-outs and VRAM; model loading and contaminated windows are not decode speed.
