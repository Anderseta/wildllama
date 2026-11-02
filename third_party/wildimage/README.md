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
