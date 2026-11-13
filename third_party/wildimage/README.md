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
It requires an existing PowerShell 7.2+, MSVC 14.39, CUDA 12.8 and CMake installation;
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

Remove `-PrepareOnly` to compile the CLI, image core and its existing custom CUDA
operations. The large GGML CUDA backend is reused unchanged, including FP8/INT8
support and the existing CPU variants. Custom image kernels target CUDA
architectures 80, 89 and 120 by default; use `-CudaArchitectures` to change that
build coverage. This is not a performance measurement or a model load.

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

`0004-workspace-aliases.patch` backports the compute-workspace, RoPE and ConvRot
parts of upstream commit `5f8c19f98ddb84e301bb8725b5260ec79e768e22`. Measurement
preserves separate external allocations and view ownership, invalidates when
their addresses change, and avoids reserving copy scratch for fresh outputs.
The GGML DLLs and WildLlama's RAM accounting stay unchanged.

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
ConvRot, projection and gated-residual operations when their output is a fresh
allocation. Existing buffers/views retain alias protection; quantization scales,
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
