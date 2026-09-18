# engine-mlx-ssd

Engine di inferenza MLX-C con **expert streaming da SSD** (MoE) per Apple Silicon.

Estensione di **engine-mlx**: stessa base MLX-C, più lo spill engine che mappa i pesi
degli expert direttamente da SSD via `mmap` + `madvise`, tenendo in RAM solo gli
expert attivi.

## Differenze da engine-mlx

| Crate | engine-mlx | engine-mlx-ssd |
|-------|-----------|----------------|
| `ops`, `mlx-ffi`, `attention`, `prefill`, `kvcache`, `serve` | ✅ | ✅ (duplicati — indipendenza totale, Opzione A) |
| `spill` | — | ✅ **Nuovo** — streaming expert da SSD |

## Crates interni

| Crate | Ruolo |
|-------|-------|
| `ops` | Operazioni MLX-C atomiche |
| `mlx-ffi` | Binding FFI MLX-C (bindgen) |
| `attention` | Attention layer MLX |
| `prefill` | Prefill GEMM MLX |
| `kvcache` | KV cache MLX |
| `spill` | **SSD expert streaming**: mmap + madvise WILLNEED/DONTNEED, expert index/tracker, speculative prefetch. Backend-agnostic. |
| `serve` | Server HTTP OpenAI-compatibile + CLI, con subcomando `--spill` |

## Architettura spill

```text
serve (CLI: server HTTP / --spill <model>)
  → spill (MmapExperts: mmap safetensors → expert index → madvise streaming)
    → ops/mlx-ffi (MLX-C forward per esecuzione esperti in RAM)
```

## Dipendenze esterne

- `nxm-shared` — tipi OpenAI, SSE, config server
- `nxm-core` — ModelManifest, EnginePlan, tokenizer
- `nxm-sampler` — strategie di sampling

## Provenienza

- Base: duplicata da `engine-mlx` (Fase 4)
- Spill: migrato da `Projects_Tmp/nxm/engines/serve-spill/spill-engine/` (core backend-agnostic)
  — i moduli MLX-gated (`forward`, `loader`, `mamba2`, `model*`) NON sono migrati
  perché dipendono da `nxm-mlx-ops` non ancora su GitHub.