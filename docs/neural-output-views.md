# Scoped neural draft output views

Patch 080 removes repeated completion calls while a neural drafter consumes
outputs from one already submitted forward. Simple and EAGLE drafting create a
new scope for each step, MTP creates one after each submitted draft batch, and
DFlash/DSpark can consume several block rows through one scope.

The first supported read still calls the existing context synchronization. A
subsequent builtin sample or host NextN read in that same scope uses the
internal synchronized getter contract. The scope ends before the next decode;
there is no persistent ready flag, cross-generation output cache, autonomous
generation loop or change to the public getters. The first-submit response hook
runs before the scope can establish completion.

Admission checks exact builtin sampler interfaces, recursively through known
chains. Top-k, temperature, distribution and greedy draft operations only touch
host candidates/history/RNG after the completed forward. Grammar, controllers,
unknown and custom samplers retain the ordinary path. Encountering one also
disables grouped reads for the remainder of that local scope; a custom sampler
cannot establish a completion assumption for subsequent rows.

Device-state transport retains its owned device rows. A requested host view
still performs any necessary lazy device download and completion. Reordering,
sampling arithmetic, RNG consumption, confidence thresholds, proposal limits,
end-token handling and carry-state ownership are unchanged.

The runtime scope contains a context pointer and two booleans on the stack. It
adds no tensor, staging buffer, retained output allocation, VRAM allowance or
model setting. This is automatic removal of proven redundant calls; existing
experimental transport and memory controls remain independent.

## Small validation

`test-neural-output` compares ordinary reads against scoped reads for exact
tokens, candidate order/index/logit/probability bits, hidden values and the
subsequent RNG stream. A fixture-only wrapper counts real scheduler backend
synchronize callbacks: three samples and two hidden reads require five ordinary
completion calls and one scoped call. Separate cases cover a new output
generation, unchanged public getter synchronization and custom-chain fallback.
No production counter or extra callback is introduced.

Its `--cuda` mode uses an untrained synthetic Qwen/MTP model under a 96 MiB
managed allocation cap. The existing optional transport is enabled only for a
separate lazy-materialization contract with a 1 MiB allowance. Actual drafting,
proposal limits, multi-sequence ordering, EOS and carry-state regressions remain
covered by the small speculative-depth and MTP-state fixtures.

`--cuda --bench` optionally measures completed-output consumption using ABBA
ordering, separately from model arithmetic. Those measurements are not trained
model tokens per second and cannot establish a universal inference gain. Real
benefit depends on backend completion overhead, draft depth and the number of
rows consumed from each forward.

One shared RTX 5060 Ti run passed the exact CPU/CUDA contracts and required
lazy-materialization checks. CUDA and CPU scheduler backends together recorded
10 ordinary completion callbacks versus two scoped callbacks, corresponding to
five versus one context completion calls. The test's managed CUDA peak was
36,759,680 bytes under its 100,663,296-byte cap.

The completed-output ABBA medians were:

| Consumed outputs | Ordinary calls | Scoped calls |
| --- | ---: | ---: |
| One sample and one hidden row | 2.44 us | 1.89 us |
| Three samples and two hidden rows | 6.45 us | 4.06 us |
| Sixteen samples | 27.30 us | 16.94 us |

These small absolute savings measure output consumption and CPU sampling on
already completed synthetic outputs. They exclude a neural forward and do not
predict trained-model throughput. The actual small speculative-depth, MTP-state,
device-state and carry-state regressions also passed separately.
