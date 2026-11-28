# Isolated image runtime

WildLlama invokes the MIT-licensed
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) runner in a
separate process. The exact source and official Windows CUDA12 archive are pinned
in `runtime-lock.json`. This revision includes the Qwen Image 2.1 Turbo custom
sigma schedule fixes introduced in October 2026.

`scripts/prepare-wildimage.ps1` verifies the official archive SHA256, extracts it
into a private directory, and copies only the NVIDIA CUDA runtime DLLs from an
already installed toolkit. It records every executable/library SHA256 and the
vendor source paths in `wildimage-runtime.json`. It never installs software or
writes downloads into C:. Do not put these ggml DLLs beside the text runner.

The payload retains upstream `stable-diffusion.cpp.txt`, `ggml.txt`, `cutlass.txt`
and the toolkit CUDA EULA when present. The NVIDIA libraries remain governed by
NVIDIA's CUDA Toolkit license. The normal Windows runtime may also use the system
MSVC runtime already installed on the computer; no redistributable is installed
by this script.

Model licenses are separate: the AtomicChat Turbo denoiser is tagged
`qwen-research`; its Uncensored Qwen3-VL encoder is tagged `apache-2.0` on the
publisher's model cards. Recipe entries pin each repository revision, expected
size and SHA256. No weights are committed or downloaded by a read-only status
request. Model installation is an explicit resumable API action.

The Windows conditioning-cache/RAM-budget build uses `scripts/build-wildimage.ps1`.
It requires an existing PowerShell 7.2+, MSVC 14.39, a CUDA toolkit and CMake installation;
it does not install tools or fetch source, dependencies or weights. Supply a checkout
at the pinned source revision with its pinned submodules already present:

```powershell
./scripts/build-wildimage.ps1 `
  -SourceDirectory G:/build/wildimage/source `
  -BaseRuntimeDirectory G:/build/wildimage/released `
  -BuildDirectory G:/build/wildimage/build `
  -Destination G:/build/wildimage/runtime `
  -PrepareOnly
```

Preparation creates import libraries from the released GGML DLL exports, checks
their recorded hashes, and runs a small CPU-only ABI probe. The imported CMake
targets propagate `GGML_MAX_NAME=160`, matching this release rather than the GGML
header's default of 64. The probe verifies the tensor structure overhead, a
130-character tensor name and one CPU tensor value before configuring the build.
The tracked conditioning, bounded-RAM and CUDA patches/headers are applied to the external source
checkout; the full vendored tree remains outside Git.

The RAM overlay exports configuration and managed host-allocation functions without
changing the existing C parameter struct layout. CLI decoding, model/weight storage,
CPU workspaces, tensors and exact conditioning payloads share the DLL's budget.
`--max-ram 0` selects bounded physical-memory Auto; explicit ceilings are never
increased. `--vram-reserve` supplies the active image profile's free GPU headroom
in GiB (default 0.25, exactly 256 MiB); explicit zero remains zero. The native
capacity check applies it after device initialization. `--stage-components on`
permits sequential phase eviction of host weights.
Failed jobs classify GPU-memory exhaustion as exit 75 and RAM exhaustion as 76;
successful jobs remain exit 0 after recovered allocation probes. The caller may
retry only GPU failures with RAM-backed GPU computation, then CPU computation.
`--disable-segmented-compute` keeps the first GPU-resident attempt monolithic;
CPU graphs retain bounded segmentation. The build replaces both native clients
and the image DLL together and advertises `features.ram_budget` only in that payload.

Remove `-PrepareOnly` to compile the CLI, image core and custom CUDA operations.
CUDA 12 can reuse the release backend. For CUDA 13, prepare the pinned archive
with `scripts/prepare-wildimage.ps1 -BuildInputsOnly`, then build `ggml-cuda`
from the pinned image `ggml` submodule using `GGML_CUDA=ON`,
`GGML_BACKEND_DL=ON`, `GGML_CUDA_GRAPHS=ON`, and `GGML_MAX_NAME=160` in
the C, C++ and CUDA compiler definitions. Pass the resulting `ggml-cuda.dll`
as `-CudaBackendPath` and set `-CudaDirectory` or `CUDA_PATH` to the same
toolkit. The packaged image backend and CUDA runtime must use the same major
version. Custom image kernels target architectures 80, 89 and 120 by default;
set `-CudaArchitectures 120` for a build specific to Blackwell GeForce.

All build outputs, temporary files, CUDA cache and the destination must be outside
C:. The destination must differ from the base runtime. Files are replaced by
unlinking the old destination entry, so an existing hard link cannot overwrite
the released payload. The new manifest records the source revision, imported DLL
hashes, patch/header hashes, every packaged file hash and the conditioning-cache
capability. This script prepares a separate payload and does not install it.

## Exact FP8 activation packing

`0003-fp8-shared-packing.patch` and `native/wildimage_fp8.{h,cu}` add explicit
packing/matmul graph nodes for contiguous, scaled E4M3 linear layers on CUDA
devices with FP8 support (compute capability 8.9+). Sibling projections share
one activation reduction and conversion. Every execution recomputes that data;
nothing is reused across denoising steps or images. Vector loads have a scalar
fallback for unaligned input views. Unsupported weights, layouts, backends,
scales and forced-F32 operations retain the ordinary linear path. LoRA, weight
scaling and bias application keep their original order.

The extension retains FP32 accumulation/output and the native global-amax E4M3
conversion, uses zero cuBLASLt workspace and declares its temporary tensors to
the existing allocator. GPU work uses the backend stream and supports graph
capture/replay. `WILDIMAGE_DISABLE_FP8_PACK=1` selects the original path for A/B
diagnostics; it is not needed for normal operation.

The pinned upstream runtime includes compute-workspace, RoPE and ConvRot
improvements from commit `5f8c19f98ddb84e301bb8725b5260ec79e768e22`. Measurement
preserves separate external allocations and view ownership, invalidates when
their addresses change, and avoids reserving copy scratch for fresh outputs.
WildLlama's RAM accounting is applied to the unified execution engine.

GPU regression tests live in `native/test_fp8.cu`. After preparation, configure
`native/CMakeLists.txt` with `ggml_DIR=<build>/imports`,
`SD_SOURCE_DIRECTORY=<prepared-source>`, Release and the matching CUDA
architecture. Build and run `test-fp8` with the isolated runtime on `PATH`.
It compares exact FP32 outputs against native FP8 multiplication, checks shared
packing, changes inputs after capture, and covers zeros, unaligned views,
realistic projection dimensions and fallback conditions. Run only with an idle,
authorized GPU. Use the full benchmark script for image pixel parity and speed;
a matrix test alone does not establish end-to-end performance.

## INT8 workspaces and final decode

`0005-int8-workspaces.patch` removes redundant output-copy scratch from INT8
projection operations; ConvRot and gated-residual improvements are included upstream.
Existing buffers/views retain alias protection; quantization scales,
CUTLASS tile selection, accumulation and model weights are unchanged. The
quantization path rejects a stale no-alias plan instead of overrunning scratch.
`test-int8-workspace` checks packed activation/scale parity, zero input, aliased
views and stale plans. `benchmark-int8` compares existing CUTLASS tile choices
without changing production dispatch; microbenchmarks do not establish image speed.

`0006-final-decode-staging.patch` finishes all runners and releases reloadable,
unpinned inactive weights after sampling (including every batch and hires pass)
and before final VAE decode. It acts only with `stage_components`; it preserves
VAE weights, blocks shared with active components, source metadata and reload
support. This also frees GPU-resident denoiser weights that ordinary disk-backed
phase eviction left allocated during decode. It does not evict between denoising
steps or cache approximate activations.

The optional `test-image-staging` target requires `SD_RUNTIME_LIBRARY` to point
at the built image import library. Invoke it explicitly with denoiser, encoder,
VAE paths and the runtime directory on an idle authorized GPU. It generates two
fixed-seed images in one context and compares decoded pixels after weight eviction
and reload.
Full-image A/B validation must also cover the existing FP8/LoRA editing path.

### Wan diffusion diagnostics

Set `WILDLLAMA_WAN_TIMINGS=1` when starting Wildllama to log weight preparation
and graph execution time for each segmented Wan diffusion evaluation. Standalone
`sd-cli` runs also need `WILDLLAMA_VIDEO_PROGRESS=1`. Both intervals use wall
time at existing segment boundaries. Graph execution can overlap weight uploads.


### Wan video execution controls

Video profiles accept `attention_backend` values `cute` (default) and `sage_fp8`.
Wan RAPID uses SageAttention2 for every diffusion self-attention and cross-attention
block when selected. Queries and keys use INT8, probability/value products use FP8,
and accumulation uses FP32. Each head sees every key. Image-to-video retains the
two independent text and reference softmaxes and sums their outputs. Both use
the same quantized queries and reuse one K/V workspace in stream order. The
Sage FP8 kernel widens its FP16-rounded output to FP32 as it writes, so it can
add the reference contribution without allocating a separate output tensor.
On compatible CUDA graphs, the FP8 packer reads the projected values using their
original strides and rounds to FP16 on load. This avoids a separate FP16 value
tensor while preserving the existing rounding and FP8 output.
Larger head groups are split to limit temporary projection storage. From
16,384 key tokens, the
key means and value scales share a tiled reduction in the existing key scratch
buffer. Text/vision encoders and VAE run at their usual precision. This mode
requires a compatible CUDA device even when weights are offloaded to RAM. An
unsupported attention operator reports the rejected constraint and tensor
layouts. SageAttention FP8 stays selected through GPU memory recovery. The
gallery records observed kernels separately from the profile preference.

The CUDA reference encoder carries the stationary early-layer activation between
eight-group graphs within a single spatial tile. Reuse requires the validated
Wan VAE layout, a bitwise-zero input tail, and a replacement activation that
fits within the size of the displaced causal history. The runner releases it at
the end of each tile. Set `WILDLLAMA_WAN_DISABLE_PREFIX_CACHE=1` for an A/B run.

The Wildllama process environment also controls decode tiling and Windows read-ahead.

* `WILDLLAMA_WAN_DECODE_TILES=adaptive` tries a larger tile on the axis that
  reduces overlapping work most. The planner checks free and configured memory,
  preserves explicit tile settings, and can shorten the temporal group when
  headroom is limited. A failed trial allocation retries the original layout.
  Spatial context changes at tile boundaries can change the decoded pixels.
* Windows Wan video generation with weight tiering enabled uses bounded file
  read-ahead by default. One low-priority worker maps small windows of the next
  GPU segment's weights and submits up to eight ranges together, with up to 32 MiB
  total per request, even when auto-fit selects buffered reads. Physical memory,
  system commit and the host budget bound each request.
  Set `WILDLLAMA_WAN_FILE_HINT=0` to disable it. With timing diagnostics enabled
  the runner reports the first accepted hint.

The offline native fixture in `tests/wan_backend_benchmark.cpp` compares full
attention using the installed GGML backend and the matching compiled media
objects. Pass the backend DLL directory, query length, key length, head count,
and a fixture output path or `-`. Use `conv` in place of the query length for a
3D convolution, with the next two arguments giving width and channel count.
Saved input and output tensors are FP32 in GGML order. One optional dispatch
override, `WILDLLAMA_WAN_NATIVE_SHAPE`, applies only to tiled Wan D128
self-attention on an exact `GPU name/CUDA driver API version/Q/K/heads` match.
Benchmark a new backend and compare generated frames before selecting an override.

## Qwen Image 2.1

The denoiser computes timestep modulation once per forward pass and shares it
across transformer blocks. Its rotary cache key describes the text segments and
image grids. On a miss the runner writes rotations into the final tensor,
reusing the entries shared by each row and column. ER-SDE moves completed
predictions into its history buffers.

When CUDA Flash Attention accepts a 128-channel head and a cached FP16 prefix,
the runner assembles keys and values at the same precision used by the original
attention operation. Other configurations use the existing attention path.
`tests/qwen_image_positions_test.cpp` checks edit layouts, circular coordinates
and the host allocation limit. `tests/qwen_cached_flash_test.cpp` compares GPU
outputs byte for byte against the original graph.

CUDA devices with the WildImage FP8 extensions can pack cached values and
fresh values directly into the Flash Attention input. One pass copies the
cached half bits and converts the fresh FP32 values. The key path retains
the backend's rotary fusion. The same fixture checks the packed half bits,
including an unaligned target and repeated graph execution.

Runtime linear LoRAs with an effective gain of exactly one pass their FP32
projection directly to the adapter sum. Other gains use the normal scaling
operation, including per-module alpha and the user's adapter strength.

On supported CUDA devices, Qwen Image 2.1 combines a scalar projection scale
with the SwiGLU activation or the gated residual addition. Each kernel reads
the projected values once and preserves the original FP32 operation order.
Reference and target tokens can use separate residual gates during editing.
`tests/qwen_scaled_projection_test.cpp` compares these kernels with the
original GPU graph, including unaligned views and graph reuse.
