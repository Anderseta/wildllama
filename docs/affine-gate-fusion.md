# Affine activation gates

The optional CUDA dense-operator path can fuse contiguous F32
`ADD -> SOFTPLUS/SIGMOID/SiLU -> MUL` into one kernel. The softplus form occurs
in Gated DeltaNet input gates, but admission uses operations and layouts rather
than architecture, model or GPU names. Existing softplus/multiply and
Conv1D/SiLU fusions remain available independently.

The new path removes the separate ADD launch and intermediate writes compared
with the existing two-launch ADD plus activation/multiply route. It uses the
original activation function, an explicitly rounded F32 sum and F32 product.
Weights, recurrent updates, sampling and requested context are unchanged.

Both bias and multiplier may be complete element-wise inputs or a contiguous
column vector broadcast over rows. Other precisions, strides, broadcasts,
observable intermediate outputs and unsafe storage aliases retain the existing
kernels. Concurrent graph regions are excluded. Exact element-wise output/input
aliases are allowed; shifted and broadcast aliases are rejected.

There are no new tensors, device buffers, staging allocations, graph arenas or
allocator lifetime extensions. Existing allocation quotas continue to apply.
The candidate uses the existing `GGML_CUDA_DENSE_OPT=1` switch; it adds no profile
field and does not enable that experiment automatically. Unsupported backends
continue executing the unchanged graph through their own operators.

`test-affine-gate --host-only` checks shape, precision, dependency, liveness,
broadcast, alias and overflow admission without loading a backend.
`--cpu` explicitly loads only the CPU backend and exercises the fallback graph.
`--cuda` compares the candidate with an unfused graph on the same backend,
including changed inputs, graph replay and live-intermediate fallbacks, under a
128 MiB managed CUDA fixture quota. The test checks actual fusion submissions and
requires bit-exact outputs. Driver and other applications' allocations are
outside that fixture quota.

Host contracts and CPU fallback checks do not validate a CUDA kernel or establish
model throughput. GPU numerical checks and measurements must pass before broader
activation; no SGLang parity or trained-model speedup is inferred from fewer
operator launches alone.

The [small validation record](benchmarks/portable-runtime-20261104.json) includes
216 CPU cases and both enabled/disabled CUDA cases, with three changed inputs
per case and exact output bits. A six-shape model-free ABBA comparison on a shared
RTX 5060 Ti showed mixed timing results, including a regression in one shape.
Those observations do not justify changing the existing opt-in default.
