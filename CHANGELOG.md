# CHANGELOG — engine-mlx

All notable changes to this crate are documented here, per `RULES.md`.

## [Unreleased]

### test
- Add unit tests for every module, organized as `crates/<crate>/tests/<module>.rs`
  (one file per module), per `RULES.md` (no inline `#[cfg(test)]`).
- Add cross-crate integration test in `crates/serve/tests/integration/` wired via
  `crates/serve/tests/integration.rs`, asserting the shared `mlx_array` type flows
  across ops/attention/kvcache/prefill and that the stubbed pipeline fails uniformly
  with the `"needs mlx-c FFI"` marker.
- Mark the real forward-pass test `#[ignore]` (requires mlx-c headers + Apple GPU),
  per `RULES.md`.
- Total: **78 tests passing, 1 ignored** (`cargo test --workspace`).
