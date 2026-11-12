# Huihui feature retest — 2026-11-11

RTX 5060 Ti 16 GiB, Ryzen 5 2600, 32 GiB RAM, Windows. Installed Go revision `80af7e9ea5931e1569abb6dc88ca3b6550ac0a4d`; native payload `6b85a649ca207ae821297e5819ff762d4024e01e`. Model: `huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF`, IQ3_S body and Q4_K output head. No weights were changed.

## Method

36 sequential native-server configurations completed 210 requests, including 132 timed requests. Short screening used two Portuguese prompts, 256 output tokens, one warmup and two timed repetitions. Confirmation used three seeds (7, 149, 987), both prompts, up to 512 output tokens, one warmup and one timed repetition per case. Separate 16,384-token raw-continuation and long-chat checks exercised physical KV and compute growth. The long chat contained 16,482 prompt tokens after formatting.

Logical context stayed 40,960, initial physical KV 8,192, Q8_0 K/V, one sequence, temperature 0.6, top-k 20, top-p 0.95, min-p 0 and repeat penalty 1. MTP used two proposals except the explicitly paired depth-four cases. Prompt checkpoints and RAM prefix caching were disabled only in the native comparison; the saved user limits were preserved.

Every accepted timed window passed the paging monitor. Guards retained 512 MiB physical-RAM/Windows-commit and 768 MiB device headroom. The old and final reference runs were close; ordinary sub-percent differences are not treated as established gains. Whole-board VRAM includes the desktop and driver, so tracked buffer sizes are reported separately.

## Short screening

Values below are means of the two timed requests, in tokens/s. These are observations for this model and workload, not universal gains.

| Variant | Prose | Code |
| --- | ---: | ---: |
| New optional switches off | 46.31 | 45.78 |
| One-token IQ3 sign switch on | 46.23 | 45.85 |
| GPU block verification, 1 MiB allowance | 46.53 | 46.04 |
| Contextual proposal shortlist | 41.58 | 46.78 |
| Shortlist plus packed proposal head | 41.57 | 46.77 |
| Four proposals, compact state | 41.75 | 40.46 |
| Four proposals, coefficient reuse | 42.23 | 40.91 |
| Repeated new-switch reference | 46.34 | 45.77 |
| Installed defaults, graph cache 128 MiB | 46.32 | 45.76 |
| Compute growth | 46.29 | 45.46 |
| Compact attention mask | 46.28 | 45.74 |
| Trim idle CUDA scratch | 46.03 | 45.40 |
| Physical KV tail reclamation | 46.22 | 45.74 |
| Paged target attention | 39.15 | 44.74 |
| Paged target and draft attention | 43.37 | 45.96 |
| Device MTP state transport | 46.23 | 45.64 |
| One-token Q4_K head | 46.33 | 45.76 |
| Selective MTP catch-up | 45.63 | 45.01 |
| IQ3 column reuse | 44.99 | 44.31 |
| GPU thinking prefilter | 45.69 | 45.10 |
| GPU probability preparation | 45.95 | 45.38 |
| Proposal-temperature tuning | 46.21 | 45.63 |
| Adaptive MTP depth | 46.84 | 49.90 |
| Additional graph cache disabled | 45.67 | 45.04 |
| Block preview disabled | 45.70 | 45.14 |
| Compact rollback disabled | 46.20 | 45.48 |
| Repeated installed-default reference | 46.28 | 45.68 |

Compact rollback, block-preview reuse and the existing compatible CUDA paths remain enabled. The one-token IQ3 sign switch showed no overall speed gain in this MTP workload. The trace of the long-chat comparison recorded the two/three-token IQ3 verification kernels; a configured one-token switch must not be reported as a measured one-token kernel hit. Coefficient reuse retained identical depth-four responses and improved those two observed means by about 1.1%; depth four remained slower than depth two.

Shortlisting changed proposals and reduced acceptance in prose. Packing did not repair that regression. GPU block verification did not establish a worthwhile whole-model gain in this screen. Those options remain off. Selective catch-up, column reuse and thinking prefilter were slower again.

## Selected local profile

The selected, persistent Huihui profile enables compute growth and compact masks, disables **additional** graph arenas, and keeps fixed two-proposal MTP. Ordinary CUDA graphs, the main scheduler, compact rollback and automatic compatible kernels remain enabled. The 128 MiB optional-arena budget is remembered in Wildgoose but its checkbox is off. This is a measured local profile choice; it does not enable optional settings indiscriminately on other models or GPUs.

| Seed / task | Previous profile | Selected profile | Adaptive selected profile |
| --- | ---: | ---: | ---: |
| prosa-7 | 46.90 | 46.06 | 46.04 |
| codigo-7 | 48.62 | 47.84 | 45.87 |
| prosa-149 | 47.02 | 46.31 | 43.93 |
| codigo-149 | 46.17 | 45.51 | 48.36 |
| prosa-987 | 51.75 | 51.01 | 50.62 |
| codigo-987 | 45.73 | 45.13 | 45.85 |

The selected profile was 1.51% slower by the geometric mean of these six ratios (range -1.79% to -1.31%). All twelve warm/timed responses, including reasoning and proposal counts, matched the previous profile exactly. The trade favors lower memory use, consistent with the requested small-speed-for-memory tradeoff.

The adaptive option looked promising in the first 256-token code request but did not generalize. In confirmation, it improved one code case and regressed others by up to 6.57% against the previous profile; it remains off. Different adaptive-depth and shortlist trajectories are expected for a fixed seed and are not themselves proof of an incorrect answer. These tests are not a broad answer-quality evaluation.

## Memory

| Quantity | Previous | Selected |
| --- | ---: | ---: |
| Target CUDA compute arena | 300.60 MiB | 142.35 MiB |
| Draft CUDA compute arena | 138.02 MiB | 122.08 MiB |
| Peak process private/committed memory | 19630.62 MiB | 14600.75 MiB |
| Peak process resident working set | 12362.24 MiB | 12080.72 MiB |
| Observed whole-board VRAM peak | 13835.22 MiB | 13585.16 MiB |

The compute arenas save **174.19 MiB** at the short-history starting size. The whole-board peak was about 250 MiB lower in the confirmation run, including removed secondary arenas and unrelated driver/desktop variation. These figures overlap and must not be added. Model GPU weights remain 11,159.69 MiB; target/draft initial KV remains 320/32 MiB and recurrent state remains 307.14 MiB.

The approximately 4–5 GiB reduction in private/committed memory is **not** 4–5 GiB of physically resident RAM. Resident working set fell about 282 MiB in the six-case comparison. Turning off only the extra arenas independently removed about 4.1 GiB of private memory in the short screen.

The source review found that `ggml_backend_sched_new` reserves host metadata using the maximum graph capacity and a worst-case split-input multiplier. Every retained scheduler repeats that reservation. The current additional-arena admission accounts for graph-result metadata and compute buffers, but not all scheduler-owned host reservations. Therefore its MiB allowance must not be described as a complete RAM/VRAM overhead cap. Disabling this optional cache mitigates the observed excess; this retest does not claim to have fixed the allocator or its accounting.

## Longer histories

| 16K-input experiment | Previous tokens/s | Selected tokens/s |
| --- | ---: | ---: |
| Long chat followed by a Python task | 41.41 | 41.28 |

The repeated long-chat response and acceptance counts matched. The raw continuation controls also matched with growth, masks and both together. Growth crossed 8,192 → 16,384 → 32,768 reserved rows while retaining the 40,960 logical context. Memory savings shrink as histories grow; neither switch changes the model weights, KV precision or maximum context.

## Reproduction and scope

The selected profile was also applied through the normal API and reloaded after
restarting the daemon. `/api/chat` confirmed MTP proposals and returned `42` for
17 + 25; `/api/ps` retained the 40,960-token context. The native log confirmed the
142.35/122.08 MiB compute arenas and compact recurrent state. Weight-layer digests
were unchanged. The lifecycle check waited for asynchronous unloading to finish
before asserting an empty model list. Test processes were stopped afterward.

Use the pinned native payload, one isolated server at a time, the settings above and the same warm/timed sequence. Set the launcher defaults explicitly when invoking `llama-server` directly: `WILDLLAMA_CUDA_GQA_Q8_VERIFY=1`, `GGML_CUDA_Q4_HEAD_WARP=1`, `WILDLLAMA_CUDA_GDN_GATHER=1`, `WILDLLAMA_CUDA_IQ3_FAST_SIGN=1`, `WILDLLAMA_GDN_REUSE_REPLAY=1`. For the selected profile additionally set `WILDLLAMA_COMPUTE_GROWTH=1`, `WILDLLAMA_COMPACT_ATTN_MASK=1`, `WILDLLAMA_GRAPH_VARIANTS_MIB=0`; keep other optional experiments off.

Prompts: “Explique em português brasileiro como organizar documentos de uma biblioteca. Descreva cada etapa em detalhes, com exemplos práticos.” and “Escreva em Python uma função para agrupar transações por mês e categoria. Inclua testes e explique as decisões em português.”
