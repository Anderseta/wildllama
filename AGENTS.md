# WildLlama — agent instructions

## Workflow

- Work on canonical `main`; create a branch only when the user requests it. Preserve Git history and push to the user's `origin`. Treat `upstream` as a reference; review upstream updates separately.
- Keep model weights, credentials, personal profiles, generated images and runtime data outside Git. Preserve model quality, context, sampling and memory limits unless the user authorizes changes.
- Prefer changes that benefit supported architectures generally. Gate backend-specific behavior by capabilities and retain a correct fallback.

## Build and validation

- Read `docs/development.md` for supported build workflows. Quick Go-only iteration against an existing native payload: `go build .`. Native changes require the corresponding CMake/backend checks.
- Reuse installed SDKs and dependencies. In this workspace, keep outputs, downloads and `TEMP`/`TMP`/`TMPDIR` on G: or F:, never C:.
- Format changed code and run focused unit/race tests. Use small fixtures first; real GPU/large-model tests require user authorization and available hardware. Coordinate loads with Wildgoose and other tests.
- Verify numerical correctness, cancellation, memory-budget behavior and compatibility where affected. Separate component measurements from real model throughput; do not claim a speedup without evidence.
- Never overwrite an executable or library in use. Verify the build's source revision and hashes before installation; preserve user configuration, model stores and unaffected native libraries. Keep experimental runtimes isolated when requested.

## Completion cleanup — required

- After validation and successful deployment, delete generated build/test directories, temporary worktrees, fixtures, benchmark binaries and obsolete runtime copies, including disposable `.build`, `build` and `target` outputs. Do not accumulate a new dated environment for every task.
- Retain concise commands, results, revisions and hashes in a small diagnostics report. Keep at most one necessary, verified rollback backup outside temporary build directories.
- Preserve source code and Git history, model weights, user data/profiles, installed runtimes, and required SDKs/dependencies. Review dirty or unmerged worktrees before removing their outputs.
- Resolve every deletion target and verify that it remains inside the intended workspace. Never recurse through junctions, symlinks or other reparse points into another directory or drive; defer targets still in use.
- Clean regenerable caches when requested or needed without deleting installed toolchains by mistake. Measure free disk space before and after and report the actual recovery.
