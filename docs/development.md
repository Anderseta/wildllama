# Development

Install prerequisites:

- [Go](https://go.dev/doc/install)
- [CMake](https://cmake.org/download/) 3.24 or newer
- C/C++ compiler: Clang on macOS, Visual Studio 2022 C++ tools on Windows, or GCC/Clang on Linux
- [Ninja](https://github.com/ninja-build/ninja/releases) in `PATH` is recommended, especially on Windows

For pure Go iteration against an existing native payload, run Ollama from the repository root:

```shell
go run . serve
```

> [!NOTE]
> Ollama includes native code compiled with CGO.  From time to time these data structures can change and CGO can get out of sync resulting in unexpected crashes.  You can force a full build of the native code by running `go clean -cache` first. 

## Native build model

The experimental GPU-exclusive MoE path has an isolated local build and numerical
regression workflow in [GPU-first inference](gpu-first-moe-inference.md) and
[memory validation](gpu-first-memory-validation.md). It uses the same pinned
llama.cpp compatibility patch stack; its laboratory listener and model manifests
are separate from production.

For a fresh checkout, or after changing native code, build from the repository root. On macOS arm64, this builds Metal inference. On all other platforms this builds CPU-only inference. It builds the Go binary at the repository root and installs the native runtime payload under `build/lib/ollama`.

```shell
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

To install into a standard prefix layout:

```shell
cmake --install build --prefix /path/to/install
```

On all platforms except macOS arm64, to build GPU backends select the backends explicitly:

```shell
cmake -B build . -DOLLAMA_LLAMA_BACKENDS="cuda_v13;vulkan"
cmake --build build --parallel 8
```

Supported backend values are `cuda_v12`, `cuda_v13`, `rocm_v7_1`, `rocm_v7_2`, `vulkan`, `cuda_jetpack5`, and `cuda_jetpack6`.

Use standard CMake architecture overrides to narrow GPU builds for local hardware:

```shell
# CUDA
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=cuda_v13 -DCMAKE_CUDA_ARCHITECTURES=native

# ROCm / HIP
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=rocm_v7_2 -DCMAKE_HIP_ARCHITECTURES=gfx1100
```

You can tune GGML build options by setting `GGML_*` values during configure. For example, to disable CUDA flash attention kernels for local debugging:

```shell
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=cuda_v12 -DGGML_CUDA_FA=OFF
```

### CUDA host waits

Start a runner with `WILDLLAMA_CUDA_SYNC_TRACE=1` to print counts and wall time
for ordinary and large-batch host waits when its CUDA backends shut down. The
trace measures the scheduler's CUDA stream fences and includes no prompt data.
Compare process CPU time separately from the trace's elapsed wait time.
The native server also groups the measured fences by its prefill, forward,
target readback, draft, draft catch-up, verification and rollback scopes.
Work submitted outside those scopes is included only in the backend totals.
The phase trace never changes the wait policy.

`WILDLLAMA_CUDA_PREFILL_BLOCKING=1` enables experimental blocking waits on
NVIDIA CUDA when a context synchronization has at least 128 evaluation tokens
queued. It records a host-wait event on the existing stream when work is still
pending. Other fences retain their existing synchronization behavior. Compare
prompt and generation timings, process CPU time, and output tokens against the
same build with the option off before enabling it for a workload.

`WILDLLAMA_CUDA_TARGET_BLOCKING=1` is a separate experiment for the target
output fence of a single-slot generation batch. The CUDA backend limits the
blocking policy to that host call. Prompt batches, the MTP draft and the
sampler keep their original synchronization paths. The phase trace reports
these waits independently of the selected policy.

For single-slot Qwen35 with Q8_0 KV, two MTP proposals and one SM120 GPU,
the launcher enables the target and prefill waits plus
`WILDLLAMA_CUDA_DRAFT_OUTPUT_BLOCKING`. The latter sleeps at the host readback
of draft output. Each switch accepts an explicit `0` override. The native
runner used directly enables these waits only when requested.

## macOS (Apple Silicon)

Additional prerequisites:

MLX Metal requires the Metal toolchain. Install [Xcode](https://developer.apple.com/xcode/) first, then:

```shell
xcodebuild -downloadComponent MetalToolchain
```

## Windows

Additional prerequisites:

- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) including the Native Desktop Workload
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/en/latest/)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_type=exe_network)
- (Optional) Vulkan GPU support
    - [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) - useful for AMD/Intel GPUs
- (Optional) MLX engine support
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)

For Ninja builds, run CMake from a Developer PowerShell/Command Prompt or another shell where the Visual Studio compiler is available.

Windows build scripts also accept extracted CUDA toolkits outside Program Files.
They validate `nvcc --version`, skip removed/runtime-only directories and consult
persisted user/machine settings when an open terminal still has an obsolete
`CUDA_PATH`. A valid process override stays first. `-CudaDirectory` is explicit
and fails if unavailable; `-CudaVersion` in the native validation build pins the
requested minor/build instead of silently switching. Release presets retain
their existing same-major fallback. Verify lookup with
`scripts/tests/test-cuda-toolkit.ps1`; it neither installs CUDA nor loads the GPU.

> Building for Vulkan requires VULKAN_SDK environment variable:
> 
> PowerShell
> ```powershell
> $env:VULKAN_SDK="C:\VulkanSDK\<version>"
> ```
> CMD
> ```cmd
> set VULKAN_SDK=C:\VulkanSDK\<version>
> ```

## Windows (ARM)

Windows ARM does not support additional acceleration libraries at this time.

## Linux

Additional prerequisites:

- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
- (Optional) Vulkan GPU support
    - [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) - useful for AMD/Intel GPUs
    - Or install via package manager: `sudo apt install vulkan-sdk` (Ubuntu/Debian) or `sudo dnf install vulkan-sdk` (Fedora/CentOS)
- (Optional) MLX engine support
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)
    - OpenBLAS/LAPACK: `sudo apt install libopenblas-dev liblapack-dev liblapacke-dev` (Ubuntu/Debian)
> [!IMPORTANT]
> Ensure prerequisites are in `PATH` before running CMake.

## MLX Engine (Optional)

The MLX engine enables running safetensor based models. On macOS arm64, MLX is enabled by default. On other platforms, MLX backends are selected with `OLLAMA_MLX_BACKENDS`.

### CUDA

Requires CUDA 13+ and [cuDNN](https://developer.nvidia.com/cudnn) 9+.

```shell
cmake -B build . -DOLLAMA_MLX_BACKENDS=cuda_v13
cmake --build build --parallel 8
```

### Local MLX source overrides

To build against a local checkout of MLX and/or MLX-C (useful for development), set environment variables before running CMake:

```shell
export OLLAMA_MLX_SOURCE=/path/to/mlx
export OLLAMA_MLX_C_SOURCE=/path/to/mlx-c
```

On macOS arm64:

```shell
OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c cmake -B build .
cmake --build build --parallel 8
```

For CUDA:

```powershell
$env:OLLAMA_MLX_SOURCE="../mlx"
$env:OLLAMA_MLX_C_SOURCE="../mlx-c"
cmake -B build . -DOLLAMA_MLX_BACKENDS=cuda_v13
cmake --build build --parallel 8
```

## Docker

```shell
docker build .
```

### ROCm

```shell
docker build --build-arg FLAVOR=rocm .
```

## Running tests

To run tests, use `go test`:

```shell
go test ./...
```

## Library detection

Ollama looks for native helper binaries and acceleration libraries in installed and local development layouts:

* `../lib/ollama` for standard installs where `ollama` is under `bin/`
* `./lib/ollama` for Windows release-style payloads and local dist output
* `.` for macOS release artifacts that colocate helpers with `ollama`
* `build/lib/ollama` and `dist/<platform>/lib/ollama` for local development builds

If the libraries are not found, Ollama will not run with any acceleration libraries.

## Hugging Face GGUF downloads

Model names allow 96 characters so valid Hub repository names are accepted.
When the Hub registry reports a sharded GGUF repository, the puller resolves
the requested quantization through the Hub file API and downloads each shard
through the existing resumable, SHA-256-verified blob downloader. Import keeps
the original shards instead of merging or copying their weights. A single
`mmproj` GGUF is included automatically when the repository provides one.

## Image model library

`GET /api/images/library` returns `models_dir`, numeric `revision`, `components`
and `pipelines` arrays. Components have opaque path-derived `id`, `name`, `path`,
`roles`, byte `size`, `ready`, `managed`, optional pinned Hugging Face `source_url`,
`profile_ids`/`profile_names`, `pipeline_ids`, `can_delete` and `delete_reason`.
Pipelines have `id`, `name`, profile references, `component_ids`, `can_remove` and
`remove_reason`. The scan reads manifests and file metadata only; it does not hash
weights, load models or contact the network. Unreferenced installed files remain
visible. Links and hidden incoming downloads are skipped. `truncated: true`
indicates an incomplete or ambiguous inventory and disables physical deletion.

`POST /api/images/library/delete` accepts `{kind: "component" | "pipeline", id,
revision, delete_files, confirm_profiles}`. A pipeline with profiles requires
`confirm_profiles: true`; removal drops those profiles and selects the first
remaining profile as default, or an empty default when none remain. With
`delete_files: false`, only the pipeline registration is removed. Physical
deletion preserves externally referenced files and components shared by another
pipeline/profile. Components can only be deleted explicitly with
`delete_files: true` when unreferenced and inside the configured model root.
Running/queued/still-stopping jobs block these operations. Unknown IDs return 404;
stale revisions, missing confirmation and unsafe references return 409.

A successful response includes `removed`, `deleted_files`, `retained_files`,
updated `config` and `library`. Deletion uses confined, individual file renames
and unlinks, never recursive directory removal. `library-delete.json` is a small
durable transaction receipt: startup restores staged files if configuration was
not published, or completes deletion after publication. Configuration/imports
and library mutations are serialized; requests prepared against an old revision
cannot enqueue generation after their profile/components have been removed.

Adding models reuses `/api/images/models/install`, `/import` (manifest references),
`/inventory` and `/copy` (verified local copies); LoRAs reuse
`/api/images/loras/install`. This library does not advertise architecture support
beyond the existing pinned runner. CPU regression checks:
`go test -race ./server -run '^TestImage' -count=1`.

Windows text-runner load logs report the minimum observed physical and commit
headroom per stage in MiB. Commit headroom measures how much additional memory
Windows can promise the process. Use page-out counters to diagnose actual
paging. Under host pressure the native loader bounds advisory file prefetch;
model mapping, context length and GPU layer placement keep their configured values.
