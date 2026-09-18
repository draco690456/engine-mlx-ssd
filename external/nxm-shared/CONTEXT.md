# nxm-shared

Tipi condivisi per tutti gli engine e applicazioni Nexum.

## Contenuto
- Tipi OpenAI-compatibili (ChatCompletion, Message, etc.)
- SSE streaming utilities
- Config parsing (TOML)
- Server traits (EngineServer, HealthCheck)

## Usato da
Tutti gli engine (crate `serve`), harness, web, tui.

## NON contiene
Logica di inferenza, kernel, operazioni tensor.