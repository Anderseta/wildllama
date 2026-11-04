# Stochastic target matching on the device

The existing bounded device verifier now accepts a fully backend-supported
terminal distribution sampler as well as greedy selection. Earlier filters must
be stateless builtins. History penalties, grammar, reasoning controllers, nested
chains, custom callbacks and an earlier token selector retain their normal path.
Distribution sampling must use the existing multi-output RNG transaction.

This optimizes only target-sample-and-match verification: compare the already
sampled target IDs with the draft and return the accepted prefix. The graph,
probabilities and random draws are unchanged. Acceptance commits one existing
distribution draw per accepted row; rejected rows do not advance the committed
stream. The EOG and bonus-token rules remain unchanged.

This does not turn p/q rejection sampling into token matching. Rejection sampling
still uses its target/draft probabilities and residual distribution on the host.
Before that path, or a device-verifier fallback, deferred candidate transfers are
queued together and completed with one fence instead of one fence per lazy row.
The final public candidate view remains available.

The optimization uses the existing verifier allowance, CUDA managed quota,
preflight release and fallback policy. It adds no sampler scratch allocation,
KV storage or new setting. An unset or zero verifier allowance keeps ordinary
transfers. Unsupported backends continue to use host verification.

## Focused validation

- `test-sampler-overlap`: CPU sampling graphs compare seeded multi-row outputs
  and subsequent RNG across zero, partial and full accepted prefixes, including
  rollback after completed but discarded work. No trained model or GPU is used.
- `test-sampler-output-view`: an independent legacy reference checks matching,
  p/q rejection, residual draws, final candidates and continued RNG with
  stochastic verification intent enabled. The host path retains one batch fence.
- `test-device-verification`: policy and unsupported-chain tests run on CPU.
  Its optional `--cuda` mode also compares actual tiny stochastic MTP results,
  checks a device-verifier hit for matching, and proves that p/q rejection keeps
  its host algorithm and materializes all deferred candidates.

These are correctness and transfer-work checks. They do not establish a trained
model tokens-per-second improvement; that requires a controlled GPU comparison.
