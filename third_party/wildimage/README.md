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
  -BaseRuntimeDirectory F:/WildLlama/lib/wildimage `
  -BuildDirectory G:/build/wildimage/build `
  -Destination G:/build/wildimage/runtime `
  -PrepareOnly
```

Preparation creates import libraries from the released GGML DLL exports, checks
their recorded hashes, and runs a small CPU-only ABI probe. The imported CMake
targets propagate `GGML_MAX_NAME=160`, matching this release rather than the GGML
header's default of 64. The probe verifies the tensor structure overhead, a
130-character tensor name and one CPU tensor value before configuring the build.
The tracked conditioning and bounded-RAM patches/headers are applied to the external source
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
