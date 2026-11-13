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

In the Wildgoose prompt editor, use `<image1>`, `<image2>` and so on in
attachment order. WildLlama maps those tags to `<Picture 1>`, `<Picture 2>`
for MiniMax H3 Image before inference. Prompts already written with the native
MiniMax tags still work. Qwen Image 2.1 uses the editor's spelling directly.
Other architectures that need conversion can register their tensor signature
and output format in `imagegen/reference_prompt.go`.

The recipe uses 1024 × 1024, six Euler steps, CFG 1 and the native six-step PDD
sigmas. It retains the downloaded weight precision, Flash Attention, staged
components, VAE tiling, RAM fallback and 256 MiB of free VRAM headroom. It does
not enable approximate Sage Attention or introduce a negative prompt. The model
requires its PDD schedule; adding arbitrary steps is not a quality upgrade.

On 16 GiB GPUs, generator and encoder run in separate phases. The encoder's GPU
weights are released before denoising. File sizes are not complete VRAM
estimates: activations and workspaces also need memory. MiniMax first keeps
weights on the GPU and lets the native allocator split work at layer boundaries.
It can evict small weight blocks when needed without restarting reference and
prompt encoding. A confirmed remaining shortage retains the bounded RAM-backed
GPU/CPU fallbacks. Resolution, references and free memory affect throughput.

For two or more references with a managed GPU budget of at most 16 GiB,
independent attention heads use a 32 MiB grouping budget. This preserves all
tokens, precision, sampling and reference resolution. Other architectures keep
their existing placement policy; recognition uses tensor signatures, not names.
Tensor loading wakes its progress reporter on completion instead of paying a
fixed polling delay for each small block.

Use `scripts/benchmark-wildimage.py --segmented
--minimax-attention-budget-mib 32` with a MiniMax profile to measure this path.
The report separates device-wide VRAM from native-process RAM on Windows:
working set is resident memory; private commit can include nonresident pages.
The benchmark omits conditioning caches and preserves logs and pixel hashes.

A controlled RTX 5060 Ti 16 GiB run (1024², two references, six steps, fixed
seed, 14500 MiB managed target) took 335 s through the previous failed GPU attempt
plus RAM retry, versus 206 s directly with segmentation. The PNG pixels were
identical. Peak native-process working set was 1.61 GiB; during sampling it was
about 1.09 GiB, while private commit reached 16.18 GiB. This is one workload,
not a general throughput guarantee. Avoiding repeated preparation is the main
gain; it does not mean the weights or required VRAM became smaller.

`0007-gguf-original-shapes.patch` restores `comfy.gguf.orig_shape.*` metadata,
including the vision encoder's five-dimensional patch embedding. It verifies
element counts, overflow, quantization block alignment and file bounds, without
rewriting or requantizing weights. The pinned native base also includes QKV
views, chunked MiniMax attention and fused SOL VAE RMSNorm/RoPE operations.

References: [native pipeline](https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/docs/minimax_h3_image.md),
[generator](https://huggingface.co/Abiray/MiniMax-H3-Image-GGUF),
[encoder](https://huggingface.co/pottokao/MiniMax-H3-TextEncoder-16GB-abliterated-GGUF).
