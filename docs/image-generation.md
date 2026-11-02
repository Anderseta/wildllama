# Local image generation

WildLlama can run a separate diffusion pipeline through its own HTTP API. The
bundled `stable-diffusion.cpp` runner runs in a child process with its own ggml
libraries. It does not replace the text runner or load image DLLs into it.
Existing chat, completion, embedding, model profiles and text memory controls
retain their existing interfaces.

## Existing local models

The Image tab can inspect a local model folder and copy selected components into
WildLlama without a network download. Copy jobs stream and verify each file,
preserve its source, and publish a bundle only after all selected files succeed.
Cancellation removes only the job's own unfinished copies. Component paths,
precision, byte counts, compatibility hints and adapter strengths are visible.

A Qwen Image 2.1 base pipeline can use its FP8 diffusion model, Comfy tensorwise
INT8/ConvRot text encoder, BF16 VAE and compatible local LoRA. An encoder containing
the matching vision tower can supply both text and reference-image conditioning
from the same physical file. Header inspection distinguishes this from a
text-only GGUF encoder that needs a separate projector.

Profiles keep their own sampling settings. The supplied local Comfy profile uses
Euler/simple, 25 steps and CFG 1; it does not inherit the Turbo sigma schedule.
LoRAs are explicit `{path, scale}` entries: zero disables an adapter. Quantized
model adapters apply at runtime rather than expanding and rewriting the base
weights. Model quantization, base/Turbo training and LoRA strength are independent
choices.

## Sequential component storage

`stage_components` uses the image runner's disk parameter backend with lazy
loading, without whole-file mmap or eager model materialization. GPU computation
remains enabled. At a component's phase boundary, the runner evicts its GPU weight
staging and releases eligible CPU parameter blocks before the next component.
Conditioning and latent results needed by later phases remain available.
The operating system may retain reclaimable filesystem cache; this mode is not a
hard whole-process RAM limit or a system-wide no-swap guarantee.

The encoder can compute on CUDA and release its weights before diffusion; forcing
the encoder to CPU is a separate placement choice. The existing per-device fitting
target and graph segmentation still bound managed image work. The image engine's
storage/cache implementation is isolated from text inference.

## Qwen Image 2.1 Turbo

The supported download recipes assemble a complete pipeline:

| Component | Source |
| --- | --- |
| Diffusion model | [AtomicChat/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF) |
| Text encoder | [AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF) |
| VAE | The Qwen Image **2.1** VAE from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) |
| Vision projector | The matching encoder repository's projector, needed for reference-image editing |

The encoder repository alone is not an image generator. Other Qwen Image/Wan VAE
files cannot be substituted for the 2.1 VAE. Recipes record exact source revisions,
component roles, lengths and SHA-256 digests. Downloads resume in the image model
directory; a pipeline becomes ready only after its required components validate.
Weights and generated files remain outside the Git repository.

The Turbo preset uses **Euler, 8 steps, CFG 1** and its matching sigma schedule.
Width and height must be multiples of 32. Advanced overrides remain configurable;
changing a distilled model's steps or guidance is not automatically an improvement.

## GPU coordination

The default **exclusive** mode closes admission for new text inference, waits for
active HTTP streams to finish, unloads resident text models and confirms their
removal before generating. It saves the image before restoring the previous text
models and their effective profiles. Restore failures are reported separately and
do not delete a successful image. Conversation messages are unaffected, although
an unloaded model's KV cache must be rebuilt on its next turn.

The queued job rechecks runner availability before unloading text. Restoration
briefly observes released GPU memory to allow delayed CUDA accounting to catch up,
then lets the normal text scheduler restore the saved profiles. Other programs
cannot extend that observation into an indefinite wait. Models requested with
`keep_alive=0` remain on demand instead of receiving an unnecessary warm reload.

Outputs receive a full PNG integrity check, including compressed pixels and
checksums, after checking the requested dimension bounds. Validation buffers are
not retained between jobs. If the runner finishes saving before a late
failure or cancellation, the validated image remains available alongside the
operation's error and any restoration warnings.

**Auto** uses the safe exclusive policy unless coexistence can be established.
Explicit **simultaneous** mode checks available GPU memory and permits matching
resident text runners. Loading or reconfiguring another text runner is refused
while an image job is active. This avoids a background embedding or chat request
unexpectedly consuming the memory admitted for the image.

| Image setting | Scope |
| --- | --- |
| `max_vram_mib` | Diffusion runner fitting/streaming target. It is **not** a hard driver-wide CUDA allocation quota. |
| `reserve_vram_mib` | Preventive free-memory margin for admission; other programs can still allocate memory afterward. |
| `offload_to_cpu`, `encoder_on_cpu`, `vae_on_cpu` | Placement of pipeline work; may reduce GPU demand at the cost of transfer or CPU time. |
| `stage_components` | Lazy disk-backed component weights with CPU/GPU release at phase boundaries, instead of retaining all component weights in host memory. |
| `vae_tiling` | Optional VAE memory reduction, separate from diffusion sampling. |
| Text `moe_ram_limit_mib` / `gpu_vram_limit_mib` | Continue to apply to their documented text-runner scopes; they do not become diffusion-process limits. |

There is no claim of a process-wide image RAM cap, system-wide swap prevention or
guaranteed fit from GGUF file sizes. Resolution, workspaces, VAE, conditioning,
projector, CPU placement and other applications all affect the actual footprint.

The runner uses its compatible Flash Attention and staged/offloaded execution.
Text-only MTP, paged KV, speculative verification and text CUDA graph patches
are not applied to a diffusion architecture. Approximate denoising skips are not
silently enabled. The optimization shared with the text product is resource
coordination, bounded work, validated configuration and visible lifecycle state.

## API

All endpoints are additive under `/api/images`:

- `GET /config`, `PATCH /config`: persisted configuration and named profiles.
- `GET /models`: installed pipeline bundles, recipes and runner availability.
- `POST /models/install`, `POST /models/import`: validated recipe installation or
  a local pipeline manifest.
- `POST /models/inventory`, `POST /models/copy`: bounded local component inspection
  and asynchronous, verified copying without modifying source files.
- `POST /jobs`, `GET /jobs`, `GET /jobs/:id`: asynchronous generation and progress.
- `POST /jobs/:id/cancel`: cancellation, child cleanup and text restoration.
- `GET /assets/:job_id/:filename`: registered generated image assets.
- `POST /assets`: upload a PNG, JPEG or WebP reference image for editing. The
  server validates and registers its own copy; the user's original is unchanged.
- `GET /loras`, `GET /loras/files?repository=owner/repo`, `POST /loras/install`:
  inspect the adapter library, preview immutable Hugging Face file metadata and
  explicitly download one selected adapter. Size and revision are shown before
  download. Checksums and LoRA layout validate before publication; downloading an
  adapter never enables it in a profile.

Generation accepts `request_id` for idempotent submission. A client that loses
the submit response can find the job with `GET /jobs?request_id=...` or cancel
through `POST /jobs/by-request/:request_id/cancel`. A restart marks unfinished jobs
interrupted instead of silently starting them again. Terminal history is bounded;
saved image assets are not removed with history trimming.

```json
{
  "profile_id": "qwen-image-2.1-turbo",
  "prompt": "A watercolor illustration of a llama reading beside a window",
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

Omitting optional generation parameters uses the selected saved profile. Reference
edits use uploaded or generated asset IDs in `reference_images`, never model-generated shell
commands or arbitrary server file paths. Progress distinguishes waiting, text
unloading, image loading, generation, saving and text restoration.

Image data directories default beside the configured model directory. Set
`WILDLLAMA_IMAGES_DIR` or choose explicit directories in the configuration to keep
models, working files and outputs on a dedicated drive. The portable Windows
bundle places its isolated engine in `lib/wildimage`.

## Wildgoose

**Integrations → Image** generates or edits images and selects a pipeline and profile, resolution, seed, sampling,
GPU policy, placement, paths and advanced settings. Assistant instructions control
when to use the image tool; the profile's prompt prefix supplies an optional visual
style. They are separate from the conversation model's system prompt. Choose an
image file as a reference and describe the requested changes to create an edited
copy, or reuse a previous result's asset ID.

LoRA downloads live in the **WildLlama → Models** screen. Image profiles select
ready adapters from that library and save individual strengths. Existing local
bundle adapters appear there too. Files with identical generic names are
distinguished by repository; manual paths remain an advanced option.

The local `gerar_imagem` tool submits a structured request after the requesting
text inference stream has ended. It pauses background indexing at its existing
safe checkpoints; WildLlama enforces actual inference admission. It returns a
reusable asset ID, original path, seed and a bounded user-only thumbnail. A
text-only conversation model receives the descriptor rather than image pixels.
Generated originals are registered as deliverables and survive temporary cleanup.

Transient submit and progress failures use bounded retries with the same request
ID and payload, so losing an acknowledgement does not start a second generation.
A saved image, its terminal operation status and text-restoration warnings are
reported separately. The chat's image tool has a one-hour minimum deadline for
the complete unload/generate/restore cycle; ordinary MCP calls keep their configured
deadlines, and explicit cancellation remains available throughout.

Small tests cover configuration, admission, cancellation, restore failures,
idempotency, downloads and asset handling without loading a large image model.
Actual image quality, peak memory and throughput must be measured with the complete
selected pipeline on an available GPU; model-free tests do not establish them.
