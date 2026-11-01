# Residual normalization fusion

Compatibility patch 061 adds a CUDA execution candidate for the contiguous F32
`ADD -> RMS_NORM -> MUL` pattern. Both the residual sum and the final weighted
normalization remain available to the graph. This is useful for residual blocks
in either dense or mixture-of-experts models; matching depends on tensor
operations and layouts, not model names.

The kernel combines residual addition with the existing RMS normalization
reduction and weight multiplication. It retains the original RMS dispatch and
reduction order and explicitly rounds the residual sum to F32 before reduction.
The residual output is written directly to its existing allocation. Compared
with the previous ADD plus fused RMSNorm/MUL kernels, eligible execution avoids
one launch and one subsequent read of the materialized residual tensor.

## Memory and fallback

There are no new backend allocations, persistent device buffers, host staging
buffers, or extended allocator lifetimes. Existing RAM and CUDA allocation
quotas therefore remain authoritative. This candidate does not change weights,
quantization, logical context, KV capacity, or speculative decoding settings.

Admission requires aligned, contiguous F32 tensors, identical input and residual
shapes, a per-column normalization weight, widths divisible by four up to
16,384, and at most 512 rows. The intermediate normalization must not have an
external consumer. Both outputs must be disjoint from each other and from all
external inputs, including constant tensors. Existing allocator aliases retain
the ordinary kernels rather than requesting a larger compute arena.

Concurrent graph regions and unsupported layouts retain the existing route.
An externally visible residual sum is supported; an externally visible
intermediate normalization deliberately prevents this fusion.

## Activation and validation

This candidate uses the existing explicit `GGML_CUDA_DENSE_OPT=1` control. Its
default remains disabled. Removing a launch is not by itself evidence of faster
end-to-end generation on every GPU or architecture.

`test-cuda-dense --host-only` checks structural admission, lifetime/output
requirements, precision, alignment, and alias rejection without initializing
CUDA. `test-cuda-dense --residual-only` performs small operator-only CUDA/CPU
comparisons with changed inputs and graph replay. It also requires actual fusion
submission for eligible enabled cases and fallback for live intermediate
outputs or unsupported widths. It loads no model weights.

Broad trained-model throughput and language-quality results are not established
by these operator tests. Keep the candidate opt-in until relevant numerical and
performance comparisons justify broader activation.
