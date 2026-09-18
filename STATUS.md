# STATUS — engine-mlx-ssd

## Stato: 🚧 Migrazione in corso (Fase 5)

**Ultimo aggiornamento**: 2026-08-23

## Cosa c'è

- [x] Workspace Cargo duplicato da engine-mlx (7 crates: ops, mlx-ffi, attention, prefill, kvcache, serve, spill)
- [x] Crate `spill` — core backend-agnostic di SSD expert streaming (mmap + madvise) migrato da serve-spill
- [x] Subcomando CLI `--spill <model>` nel crate serve (index + prefetch/release esperti)
- [x] CONTEXT.md aggiornato con architettura spill

## Cosa manca

- [ ] Verificare cargo check dell'intero workspace
- [ ] Commit + push su GitHub
- [ ] CI GitHub Actions
- [ ] Integrare i moduli MLX-gated dello spill engine (forward, loader, mamba2, model*) — bloccati da dipendenza `nxm-mlx-ops` non migrata

## Spill engine — migrazione parziale (core portable)

| Modulo | Stato | Note |
|--------|-------|------|
| `mmap_experts` | ✅ | mmap safetensors, expert index, madvise streaming |
| `expert_index` | ✅ | byte-range per expert per layer |
| `expert_tracker` | ✅ | tracking recent usage per layer |
| `speculative_prefetch` | ✅ | prefetch lookahead |
| `engine_core` | ✅ | traits Engine/Model, Tensor |
| `engine_trait` | ✅ | SpillEngineTrait |
| `interface` | ✅ | ExpertStore trait |
| `types` | ✅ | tipi condivisi |
| `config` | ✅ | SpillConfig |
| `forward`, `loader*`, `mamba2`, `model*` | ⏸️ | NON migrati — MLX-gated, dipendono da `nxm-mlx-ops` (unmigrato) |

## Prossimi step

1. `cargo check` — risolvere errori
2. Commit + push su GitHub
3. Quando `nxm-mlx-ops`/nxm-operations saranno migrati, portare anche i moduli MLX-gated del spill engine