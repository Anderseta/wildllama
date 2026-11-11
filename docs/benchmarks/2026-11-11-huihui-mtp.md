# Huihui MTP efficiency — RTX 5060 Ti

Local measurement on 2026-11-11, Ryzen 5 2600, 32 GiB system RAM and RTX 5060 Ti
16 GiB. Model: Huihui Qwen3.8 27B IQ3_S, Q4_K output head, original weights.
Configuration: 40960 logical context, 8192 initial physical KV cells, Q8_0 target
KV, one sequence, MTP depth 2, temperature 0.6, top-k 20, top-p 0.95, seed 149.
Two Portuguese prompts (prose and code), 256 generated tokens, one warmup and
two timed repeats per prompt. These are short-prompt generation measurements,
not a full-window prefill or general answer-quality benchmark.

Prompt checkpoints and the RAM prefix cache were disabled only for the controlled
comparison. Saved user context, sampling, cache budgets and MTP depth were not
changed. Memory monitoring retained 512 MiB physical-RAM/commit and 768 MiB GPU
free-memory guards. Warmup was excluded, and timed intervals marked as paging
were discarded/retried. All reported timed intervals passed those checks.

## End-to-end results

| Variant | Prose tokens/s | Code tokens/s | Decision |
| --- | ---: | ---: | --- |
| Initial reference, new switches off | 44.29 | 43.61 | Reference |
| Reused block preview | 45.45 | 45.05 | Enabled |
| Preview + one-token Q4_K kernel | 45.40 | 44.61 | Optional, off |
| Preview + adaptive proposal temperature | 45.00 | 44.57 | Optional, off |
| Preview + contextual shortlist, 4096 | 40.17 | 45.53 | Optional, off |
| Preview + contextual shortlist, 16384 | 41.43 | 41.07 | Not selected |
| Preview + GPU probabilities | 45.04 | 44.30 | Optional, off |
| Repeated reference, new switches off | 45.21 | 44.52 | Drift control |
| Final production-layout payload, defaults | 45.34 | 44.97 | Selected |

The reference itself improved about 2.1% during the session. Therefore the
initial 2.6–3.3% preview delta must not be presented as a reliable overall speedup.
The final payload was only 0.3–1.0% faster than the repeated reference; that is
small relative to machine variation. The measured CPU verifier component did
less work: 360 fixture blocks took 8079 microseconds on the old path and 4063
microseconds with reuse. This is a small-fixture component result, not a 2x model
speedup. The selected path preserves tested tokens, acceptance counts and RNG
behavior, with no additional GPU buffers.

The shortlist changes proposals, so its text can differ under the same seed.
Its restricted proposal distribution is still passed to corrected verification.
Reduced acceptance outweighed head savings in prose, and the larger shortlist
also regressed code. Proposal temperature tuning did not select another arm in
these requests. Neither experiment is enabled by default.

## Memory and longer speculation

| Depth 4 layout | Prose tokens/s | Code tokens/s | Recurrent GPU buffer |
| --- | ---: | ---: | ---: |
| Full states | 41.00 | 40.24 | 748.12 MiB |
| Extended compact rollback | 41.81 | 40.21 | 322.90 MiB |

Compact rollback saves **425.22 MiB** at depth 4, with identical tested output and
acceptance counts. It improved prose about 2%, while code was effectively equal
(-0.08%). The compatible extension is enabled by default, but the user's depth
remains 2: both depth-4 variants were slower than depth 2. At depth 3 the state
geometry saves 283.48 MiB; that depth was tested in the kernel fixture, not in
this full-model speed comparison.

At depth 2, the reference and final payload reported identical tracked buffers:
11159.69 MiB model weights, 320 + 32 MiB target/draft KV, 307.14 MiB recurrent
state, and 300.60 + 138.02 MiB compute arenas. Whole-board peaks were 14450.54
and 14141.95 MiB, with different idle usage. That difference is **not** evidence
that the preview saved 309 MiB; desktop, driver and unrelated allocations are
included. No default was selected that adds GPU storage.

## Correctness and rejected kernels

- 360 deterministic preview rounds plus penalties, continued RNG and active/lazy
  grammar checks passed. Final CPU and CUDA sampler suites passed.
- GPU probability fixtures checked 24 rows; maximum observed absolute probability
  difference from CPU was 1.49e-8. This optional path is not bit-identical.
- Q4_K projection fixtures checked 18 cases bitwise. The new one-token kernel
  improved its isolated component only about 2.3%, without a consistent full-model gain.
- 75 extended recurrent rollback cases passed bitwise; model fixtures also checked
  12 rollbacks, checkpoint bytes, cross-layout restoration and explicit disabling.
- Indexed proposal projections preserved selected logit bits in 16 CUDA graph
  replays using bounded slices of the real head weights (under 32 MiB each).
- GDN/normalization fusion regressed its isolated component by roughly 40–46%.
  Static unrolling helped one token slightly but regressed two/three. Both were
  excluded from production.

See [implementation and controls](../mtp-decode-efficiency.md) for compatibility,
fallbacks and the distinction from full GPU verification or tree speculation.
