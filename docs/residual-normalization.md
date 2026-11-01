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
shapes (or a column-broadcast bias), a per-column normalization weight, widths divisible by four up to
16,384, and at most 512 rows. The intermediate normalization must not have an
external consumer. Both outputs must be disjoint from each other. The normalized
output remains disjoint from all external inputs; the residual output can share
the exact element-wise input allocation under patch 065's register-retained
vector dispatch. Shifted aliases and broadcast parameter overwrites retain the
ordinary kernels rather than requesting a larger compute arena.

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

## Additional operation patterns

Patch 065 broadens operation-based coverage without new persistent allocations:

- Residual RMS normalization also accepts a contiguous per-column bias. Exact
  element-wise ADD aliases can reuse the existing allocation: the vector kernel
  retains the separately rounded sum in registers for normalization. The actual
  dispatch must confirm vector rows before submitting any GPU work; otherwise
  ordinary kernels run. Width and alignment guards prevent a two-pass scalar
  implementation from being selected under that alias contract.
- Classic LayerNorm followed by column-wise multiplication and addition runs
  as one kernel. It reuses the original LayerNorm dispatch, reduction order,
  mean, variance and normalization arithmetic. Explicit F32 multiplication and
  addition preserve the original two affine node boundaries. Weights and biases
  cannot alias the output; an exact element-wise input/output alias is safe.
- F32 bias/residual ADD followed by SiLU, GELU, erf-based GELU or ReLU runs as one
  element-wise kernel using the original activation functions and a separately
  rounded F32 sum. Column bias uses fast integer indexing. The existing successful
  SiLU/MUL fusion retains priority when it can already execute, avoiding an
  unnecessary substitution of equivalent launch counts.

These candidates match tensor operations instead of installed model names and
can apply to compatible dense and MoE blocks. Live elided intermediates,
unsupported layouts, unsafe aliases and concurrent graph regions retain the
existing execution. No allocator lifetime extension forces admission, and no
quality, quantization or memory setting changes automatically.

The expanded host-only tests include width/row dispatch boundaries, column
broadcasts, exact/shifted/broadcast aliases, live intermediate outputs, precision
and stride guards. `test-cuda-dense --affine-only` prepares small LayerNorm and
four activation operator comparisons with replay and actual-path counters.
Adding these tests does not imply GPU execution while the GPU is occupied.
All additional fusions remain behind the existing explicit dense-opt switch
until numerical and performance comparisons justify broader activation.
