# nexum_core — Layer 1 Trait Contracts

Pure trait definitions for the Nexum inference framework. Platform-independent,
zero OS dependencies. Compiles on Linux, macOS, Windows.

## Crates

| Crate | Purpose |
|-------|---------|
| `nxm-ops-core` | Tensor types + operations backend trait |
| `nxm-cache-core` | KV cache + expert store traits |
| `nxm-attention-core` | Attention mechanism traits |
| `nxm-model-core` | Model, engine, architecture traits |
| `nxm-tokenizer` | Tokenizer trait + HuggingFace wrapper |

## Architecture

```
nexum_core (Layer 1 — trait contracts, quasi mai cambiano)
    ↑
Layer 2 (backend implementations: ops-cpu, ops-mlx, ops-metal, cache-memory, cache-ssd, ...)
    ↑
Layer 3 (model families: model-llama, model-mixtral, model-phi, ...)
    ↑
Layer 4 (server engines: server-cpu, server-mlx, server-metal, ...)
```

## Dependency Graph

```
nxm-ops-core         (zero external deps, only thiserror)
nxm-cache-core       → nxm-ops-core (for Tensor/DType)
nxm-attention-core   → nxm-ops-core + nxm-cache-core
nxm-model-core       → nxm-ops-core + nxm-attention-core + nxm-cache-core
nxm-tokenizer        → thiserror (standalone)
```

## Principles

1. **Trait-only** — No implementations, no concrete backends
2. **Platform-independent** — No `cfg(target_os)`, compiles everywhere
3. **Stable contracts** — Semver strict, changes are rare and always major
4. **Zero-cost abstraction** — Traits designed for static dispatch when needed
5. **Minimal dependencies** — Only `thiserror` in workspace

## Related Projects

- `nexum_core_metal` — Metal/GPU trait contracts (macOS only, depends on this)
- `nxm-ops-cpu` / `nxm-ops-mlx` / `nxm-ops-metal` — Layer 2 implementations
- `nexum-server-mlx` / `nexum-server-cpu` — Layer 4 server engines
