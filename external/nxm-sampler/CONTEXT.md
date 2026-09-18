# nxm-sampler

Strategie di sampling per inferenza LLM (top-k, top-p, temperature, repetition penalty).

## Contenuto
- Sampler GPU-resident (nessun CPU readback)
- Metal dispatch per operazioni di sampling
- Configurazione unificata (SamplerConfig)

## Usato da
Tutti gli engine (serve crates), harness.

## Dipendenze esterne
- objc2-metal (Metal compute)