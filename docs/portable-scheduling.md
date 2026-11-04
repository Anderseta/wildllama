# Capability-based mixed prompt scheduling

The existing explicit `WILDLLAMA_ADAPTIVE_PREFILL=1` experiment can now consider
compatible neural/ngram verification batches and plain recurrent/hybrid memory,
in addition to ordinary unified KV generation. It does not change a saved
profile, enable concurrency, alter proposals or sampling, reduce context, or
raise any RAM/VRAM budget. The default remains unchanged.

## Admission and sequence ownership

The native memory proof accepts exact plain KV, recurrent and hybrid classes.
Attention must use one unified stream with enough sequence slots. Recurrent
storage must provide independently allocated rollback cursors and cells for all
configured sequences. Derived/indexed memories, SWA, PLE, split streams, tensor
overrides, weight-streaming providers and unsupported placement retain ordinary
admission. The server additionally excludes media in either the incoming or
cached prompt, parent/child requests, LoRA, embeddings, constrained grammar and
audio. Text M-RoPE uses the server's existing aligned text coordinates.

Each existing generating sequence's token/proposal span must be contiguous,
disjoint, fully accounted for and wholly contained in the first configured
logical decode batch. The controller only limits the additional prompt rows
after that span; it never splits, reorders, pads or truncates a verification.
Changed sequence identities or proposal shapes discard the previous latency
estimate even when the total row count stays equal. Timings use completed
target decode calls; draft preparation and host publication are not mislabeled
as target GPU time. The original throughput floor and at-most-one consecutive
prompt deferral retain progress under contention.

For a neural draft, both contexts must prove the required memory/output
capabilities. Both target and draft catch-up use the existing graph cache in
reuse-only mode during a contention episode. Separate reservation observations
detect unexpected draft growth and disable adaptation. Every scope restores its
previous mode, including on error. Ordinary allocation quotas still govern
mandatory compute storage; the feature adds only bounded CPU accounting, not
a device cache, KV capacity or workspace.

The Go wrapper's architecture-specific single-request restrictions are preserved.
A native memory capability and a small synthetic regression do not establish
that every complete model family can safely run parallel requests. In particular,
this change does not silently increase Qwen hybrid concurrency.

## Validation and interpretation

* `test-prefill-batch-contract` is a standalone CPU policy/ownership test:
  intact spans, gaps/overlaps, overflow, workload changes, cancellation identities
  and prompt fairness.
* `test-hybrid-prefix` additionally executes a nonzero tiny hybrid model with
  two sequences, four-row verification-shaped generation and interleaved prompt
  slices. It compares same-batch logits and serialized states exactly, checks
  continuations against independent sequence executions and isolates cancellation.
* `integration/native_portable_scheduling_smoke.py` runs an explicit existing
  untrained fixture of at most 4 MiB as target and neural draft on CPU. It checks
  actual adaptation markers, intact speculative rows, bounded target/draft graph
  peaks, exact forced printable SSE/stop semantics, partial-prompt cancellation
  and subsequent reuse. No model/server is started without `--execute`.

These are small correctness and resource contracts. They do not demonstrate
trained-model tokens/s, task quality, universal model coverage or SGLang parity.
Single isolated prompts keep ordinary admission; the intended benefit is lower
interference when eligible requests share an already configured runner.
