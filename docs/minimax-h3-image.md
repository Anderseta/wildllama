# MiniMax H3 Image

The isolated image runner supports MiniMax-H3-Image generation and reference
editing alongside Qwen Image and Noct. Text inference uses separate libraries.

The image catalog includes `minimax-h3-image-q5-abliterated-12.5g`, with pinned
revisions and resumable downloads for these three files:

| Role | Hugging Face file |
| --- | --- |
| Generator | `Abiray/MiniMax-H3-Image-GGUF/MiniMax-H3-Image-Pruned-Q5_K_M.gguf` |
| Text and vision encoder | `pottokao/MiniMax-H3-TextEncoder-16GB-abliterated-GGUF/MiniMax-H3-TextEncoder-Qwen3VL-32B-abliterated-12.5G_vis.gguf` |
| Image VAE | `Comfy-Org/MiniMax-H3-Image/vae/minimax_h3_image_vae_fp16.safetensors` |

The encoder already includes vision; no separate projector is needed. The
MiniMax video VAE and the Qwen Image VAE are not compatible replacements.
Existing profiles are preserved. Catalog/status requests never download weights.

The recipe uses 1024 × 1024, six Euler steps, CFG 1 and the native six-step PDD
sigmas. It retains the downloaded weight precision, Flash Attention, staged
components, VAE tiling, RAM fallback and 256 MiB of free VRAM headroom. It does
not enable approximate Sage Attention or introduce a negative prompt. The model
requires its PDD schedule; adding arbitrary steps is not a quality upgrade.

On 16 GiB GPUs, generator and encoder must run in separate phases. Their file
sizes are not complete VRAM estimates: activations and workspaces also need
memory. Generation first attempts GPU residency; a confirmed memory shortage
retries with bounded RAM-backed GPU computation. Resolution, references and
available system memory affect throughput. Full-model timing and peak memory
must be measured after downloading the chosen weights.

`0007-gguf-original-shapes.patch` restores `comfy.gguf.orig_shape.*` metadata,
including the vision encoder's five-dimensional patch embedding. It verifies
element counts, overflow, quantization block alignment and file bounds, without
rewriting or requantizing weights. The pinned native base also includes QKV
views, chunked MiniMax attention and fused SOL VAE RMSNorm/RoPE operations.

References: [native pipeline](https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/docs/minimax_h3_image.md),
[generator](https://huggingface.co/Abiray/MiniMax-H3-Image-GGUF),
[encoder](https://huggingface.co/pottokao/MiniMax-H3-TextEncoder-16GB-abliterated-GGUF).
