# nxm-core

Core types per tutti gli engine e applicazioni Nexum.

## Contenuto (crates)
- `nxm-ops-core` — trait operazioni atomiche (matmul, softmax, etc.)
- `nxm-cache-core` — KV cache trait + tipi base
- `nxm-attention-core` — attention layer trait (GQA, RoPE)
- `nxm-model-core` — ModelManifest, EnginePlan, model-plan
- `nxm-tokenizer` — tokenizer trait + implementazioni
- `model-plan` — introspezione modelli → ModelManifest (binario nxm-modelplan)

## Usato da
Tutti gli engine (engine-metal, engine-mlx, etc.), harness, memory.

## Dipendenze esterne
- nxm-shared (via git dep)