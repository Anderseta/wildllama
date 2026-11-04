# Measured execution-plan retention

The native context now chooses which of its existing optional exact graph plans
to retain using observed preparation cost, bounded use frequency, memory charge
and aging. A recurring expensive plan is protected from a one-off shape. Missing
measurements retain the original LRU policy. Full topology, input and backend
compatibility checks remain mandatory before every actual reuse.

The policy is backend-independent and context-owned: it does not depend on a
GPU brand or model-name list. It learns from the work the caller requested and
resets with graph invalidation. It never runs duplicate inference or starts
calibration during a prompt. There are 16 observation slots and at most 2 KiB of
additional host metadata; no GPU buffer or device timing event is introduced.

Measurements are host-visible graph construction/allocation wall times. Existing
allocator waits can contribute, so they are not pure CPU time or GPU kernel
timings. A bounded average and separate timing-confidence aging prevent a stale
startup sample from permanently protecting a plan. This is adaptive graph-plan
retention, not a general automatic kernel/fusion benchmarking system; the
explicit CUDA matrix calibrator remains independent.

The existing positive `WILDLLAMA_GRAPH_VARIANTS_MIB` allowance is still required.
Its value and the managed device quota are unchanged. Disabled configurations
keep their original execution path. The planner never expands context, changes
weights, sampling or quantization, or increases the allowed number of arenas.

`test-execution-plan` checks hot-plan protection, measured cost, aging, bounded
history, invalidation and missing/invalid measurement fallback without a model
or GPU. The existing tiny graph-variant tests check actual native graph reuse,
unchanged logits/seeded tokens and low-budget fallback. Component observations
do not establish a trained-model tokens/s gain or parity with SGLang.
