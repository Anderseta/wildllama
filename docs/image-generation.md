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

## RAM backing and sequential components

New image profiles enable `offload_to_cpu` as a fallback policy. A GPU-resident
attempt runs first. Only a classified GPU memory failure moves weights to RAM
backing while retaining GPU computation. Inactive host weights can be evicted and
loaded again from their original files under pressure. `stage_components` also
releases an inactive component's weights at the phase boundary; it can remain
enabled together with RAM backing. Conditioning, references and latents required
by the next phase remain available. Explicit `encoder_on_cpu` and `vae_on_cpu`
choose CPU computation independently. With RAM offload disabled, sequential mode
retains its previous lazy disk-backed behavior.

The active image profile supplies both memory settings. `max_ram_mib=0` selects
bounded Auto: available physical RAM minus the larger of 1 GiB or 5% of total
physical RAM. An explicit RAM value is a ceiling and is only reduced if that
available allowance is smaller. Explicit RAM/VRAM values are never increased or
borrowed from a text profile. Admission may reduce the job's temporary GPU target
without changing the saved profile.

The native RAM ledger admits managed tensor storage, reference decoding, host
weights, CPU workspaces/cache and conditioning payloads before allocation. Copy,
growth and scratch buffers reserve their overlapping lifetimes. Graph cuts run
within either GPU or CPU capacity; inactive reloadable weights are reclaimed first.
This is a managed allocation budget, not an operating-system process RSS quota:
library/driver overhead, small metadata and reclaimable filesystem cache remain
outside its scope. Unknown/insufficient automatic RAM never becomes unlimited.

The image profile's free-VRAM headroom defaults to 256 MiB. This is a free-memory
margin, not a 256 MiB allocation ceiling. When RAM fallback is enabled, an
unavailable GPU can select CPU initially. A classified GPU memory failure can
advance through clean contexts: GPU-resident weights, RAM-backed weights with GPU
computation, then CPU computation if GPU working data still cannot fit. Every
attempt preserves references/order, seed, model precision and sampling settings. A completed PNG,
cancellation, RAM-budget failure or unrelated error never triggers that retry.
CPU execution can be much slower. If the working data cannot fit either allowance,
the job reports a memory error instead of exceeding its managed budget.

### Exact conditioning cache on SSD

The private patched runner can persist prompt-encoder outputs using the
`conditioning_cache_mib` budget (default 256 MiB; 0 disables it). It stores the
original float32/int32 tensor bits, without quantization or approximate reuse.
The cache has checksummed entries, bounded allocations, atomic publication,
cross-process locking and LRU eviction. Unsupported conditioning structures or
cache/filesystem failures use the normal encoder path. No persistent GPU tensor
or RAM model cache is added.

Keys include the native runtime manifest, effective profile, model file identities,
encoder/LoRA/reference content hashes and every actual conditioning parameter.
Reliable filesystem change times invalidate weight-digest memoization, including
same-size edits with a restored mtime. The first unchanged encoder signature may
need a full file read; its real progress is reported separately from generation.
Sampling seeds, steps, CFG, sampler, sigmas and scheduler do not invalidate
deterministic conditioning; model, adapter, encoder and backend changes still do.
The supported lazy
disk parameter backend can avoid encoder computation on a hit; actual loading
time and peak memory still depend on the selected model and backend.

`scripts/build-wildimage.ps1` applies the maintained patch to the pinned source and
uses the released GGML DLLs after an ABI/name-length probe. Only image-core/CLI
code is rebuilt; the text engine and the released GGML CUDA kernels are separate.

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

Each profile can enable `preserve_reference_size` (default `false`). With one or
more references, the first original's verified pixel dimensions select the nearest
supported output size. Each axis rounds to a multiple of 32 within 64–4096; if the
original exceeds 4096 on either axis, both axes first scale proportionally to fit.
For example, 1920×1080 selects 1920×1088, and 8192×4096 selects 4096×2048. The
native pipeline resamples its working tensors normally: the uploaded original is
not modified, copied, padded or cropped. Width/height request overrides are ignored
only while this preference and a reference are both present. Without a reference,
the profile defaults and normal request overrides retain their existing behavior.
Job generation summaries record the effective native width/height and the profile
preference; retries with the same request ID preserve the original job snapshot.

## GPU coordination

The default **exclusive** mode closes admission for new text inference, waits for
active HTTP streams to finish, unloads resident text models and confirms their
removal before generating. It saves the image before restoring the previous text
models and their effective profiles. Restore failures are reported separately and
do not delete a successful image. Conversation messages are unaffected, although
an unloaded model's KV cache must be rebuilt on its next turn.

Requests may set `count` from 1 to 8. Images run sequentially under one GPU lease,
with one unload/restore cycle for text; results are published durably one by one.
`results` contains the ordered batch, while `result` remains its first output for
older clients. Explicit seeds increment safely through the batch. Later failures
or cancellation preserve already completed images and still attempt restoration.
Cancelling a queued request records its terminal state immediately and releases
its queue slot. Surviving requests retain FIFO order; a cancelled request does
not unload or restore a text model when the worker reaches it.

An explicit `defer_text_restore` request can defer a successful full batch's text
warmup until the next text request. Failure or early cancellation retains normal
restoration. Completion and late cancellation have a single synchronization point
so a committed deferred success cannot become a cancelled job without restoration.
Jobs retain chat IDs, parent asset IDs and effective generation summaries for
recovery and reference-image follow-ups; API queries can filter by session ID.

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
| `max_ram_mib` | Active image profile's managed host allocation ceiling; 0 uses bounded Auto with a physical-memory reserve. |
| `offload_to_cpu` | Permit RAM-backed weights with GPU staging, then CPU computation only after classified GPU memory failures. The initial attempt prefers GPU residency. Defaults to true for new profiles. |
| `encoder_on_cpu`, `vae_on_cpu` | Explicit CPU computation for those components. |
| `stage_components` | Release inactive component weights between phases. Works with RAM backing; without RAM offload, preserves lazy disk-backed execution. |
| `vae_tiling` | Optional VAE memory reduction, separate from diffusion sampling. |
| `preserve_reference_size` | Choose the nearest supported canvas from the first reference's pixel dimensions; defaults to `false` and leaves original files unchanged. |
| Text `moe_ram_limit_mib` / `gpu_vram_limit_mib` | Continue to apply to their documented text-runner scopes; they do not become diffusion-process limits. |

There is no claim of a process-wide RSS cap, system-wide swap prevention or
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

Reference uploads may include `request_id` and `session_id` query parameters.
The original bytes are retained unchanged. An exact retry returns the same asset;
reusing the ID with another body, media type or session returns a conflict. The
`image_upload_idempotency` capability advertises this additive behavior. Missing
or modified saved originals are reported rather than silently replaced.

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
