//! # engine-mlx-serve
//!
//! MLX-C inference server — OpenAI-compatible HTTP API.
//!
//! **NOTE**: This is a minimal stub for workspace structure.
//! Full MLX-C integration requires complete MLX-C FFI implementation.

use anyhow::Result;
use axum::{Router, routing::get, routing::post, Json};
use engine_mlx_ops::MlxCtx;
use nxm_shared::types::{ChatRequest, ChatResponse, ChatMessage};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // `nxm-engine-mlx --spill <model>` — index MoE experts via SSD streaming
    // (mapped from the engine-mlx-spill crate). Skips the HTTP server.
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--spill") {
        let model = args
            .get(pos + 1)
            .ok_or_else(|| anyhow::anyhow!("--spill <model-path> required"))?;
        info!("Running SSD spill index for {model}");
        return run_spill_index(model);
    }

    info!("Starting nxm-engine-mlx (stub)");
    
    // Initialize MLX context
    let _ctx = MlxCtx::gpu();
    info!("MLX GPU context initialized");

    // Health check endpoint
    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/v1/chat/completions", post(chat_completions));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    info!("Server listening on http://127.0.0.1:8080");
    axum::serve(listener, app).await?;
    
    Ok(())
}

/// Index and stream MoE experts from SSD via mmap + madvise.
fn run_spill_index(model_path: &str) -> Result<()> {
    use engine_mlx_spill::{MmapExperts, SpillConfig};

    let config = SpillConfig::default();
    let store = MmapExperts::open(std::path::Path::new(model_path), &config)?;

    println!("=== NXM SSD Spill Index ===");
    println!("Model: {model_path}");
    println!("Hidden size: {}", store.hidden_size());
    println!("Intermediate size: {}", store.intermediate_size());
    println!("MoE layers: {}", store.num_moe_layers());
    println!();

    // Exercise the streaming path: walk each MoE layer, prefetch + release
    // the first expert into/out of RAM via madvise.
    let mut touched = 0usize;
    for &layer in store.moe_layer_indices() {
        store.prefetch_experts(layer, &[0])?;
        store.release_experts(layer, &[0])?;
        touched += store.index_len(layer);
    }

    println!("Streamed {touched} experts across {} MoE layers", store.num_moe_layers());
    println!("=== Done ===");
    Ok(())
}

async fn chat_completions(
    Json(req): Json<ChatRequest>
) -> Json<ChatResponse> {
    // TODO: Implement actual inference
    Json(ChatResponse {
        id: "cmpl-mlx".to_string(),
        object: "chat.completion",
        created: nxm_shared::types::now_unix(),
        model: req.model,
        choices: vec![nxm_shared::types::Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: "MLX engine not yet implemented (stub)".to_string(),
            },
            finish_reason: "stop".to_string(),
        }],
        usage: nxm_shared::types::Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    })
}