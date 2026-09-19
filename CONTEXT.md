# CONTEXT — engine-mlx-ssd

## What this is
An independent (no-external-repo) Rust workspace: an MLX-C inference engine for
Apple Silicon, with **SSD expert streaming (MoE)** via the dedicated `spill` crate.

## Workspace
8 internal crates + 1 spill crate (`crates/`):
`ops` · `mlx-ffi` · `attention` · `prefill` · `kvcache` · `serve` · `modelplan` ·
`tokenizer` · `spill`.

`Cargo.toml` (workspace): `resolver = "2"`, in-repo path crates only, `mlx-sys = "0.2"`
(crates.io). **No git dependencies** — fully self-contained ("tutto dentro").

## Build / run
```sh
cargo check --workspace --all-targets        # verify (macOS arm64)
cargo build -p engine-mlx-serve              # build the server binary `nxm-engine-mlx`
cargo build -p engine-mlx-serve --features mlx  # enable real MLX-C ops
# Serve:
ENGINE_MLX_MODEL=~/models/... nxm-engine-mlx          # OpenAI-compatible HTTP on :11435
# SSD spill index (no GPU needed):
nxm-engine-mlx --spill <path/to/model>
```

## Independence model
- `modelplan` + `tokenizer` are **inlined** top-level crates (pure Rust) — not pulled from
  any external repo / git dependency. This replaces the older `external/` vendored-snapshot
  approach (nxm-shared / nxm-core / nxm-sampler) used by the original stub: the inlined
  crates are themselves part of this repo, so "everything is inside".

## Cross-engine notes (what does NOT apply here)
This is an **MLX-C** engine. Compute runs through `mlx-sys`/`libmlx` (the MLX framework),
NOT raw Metal. Therefore:
- The `engine-metal-ssd` / `engine-metal` Metal-shader improvements (the q8 compute/serve
  path, `load_vector` trick, the Q8 load-shape race-fix, fused kernels) are **Metal-specific**
  and do not transfer to this MLX engine.
- 8-bit (Q8) compute here is **native**: `engine_mlx_ffi::MlxCtx::quantized_matmul` accepts a
  `bits`/`group_size`/`mode` (affine/mxfp4), and `ops::quant::QuantWeights::with_quant(..., bits=8)`
  routes through it. No custom q8.metal kernel is required.
- For the Metal engine, see `engine-metal-ssd` (draco690456/engine-metal-ssd).

## Git
- Remote: `https://github.com/draco690456/engine-mlx-ssd.git` (public).

## Recovery note
`engine-mlx-ssd` was a stub skeleton. It has been recovered by syncing the real
`engine-mlx` implementation into it while preserving ssD's `spill` crate (SSD MoE streaming).
See `STATUS.md` for the commit history and decisions. The old stub tree is preserved as the
parent commit `6b6b545` (non-destructive).
