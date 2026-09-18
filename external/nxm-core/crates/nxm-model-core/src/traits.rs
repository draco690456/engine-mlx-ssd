//! Core model and engine traits.
//!
//! Hierarchy:
//! - `InferenceEngine` — server sees this (includes tokenization)
//! - `Engine` — loads models, checks availability
//! - `Model` — raw forward/step (no tokenization)
//! - `BackendEngine` — pattern for Metal/MLX/SSD backend engines
//! - `SpecModel` — speculative decoding draft model

use std::path::Path;

use nxm_ops_core::Tensor;

use crate::architecture::Architecture;
use crate::config::ModelConfig;
use crate::error::Result;
use crate::types::ChatMsg;

/// The server-facing inference contract.
///
/// This is THE trait boundary between server and all engines.
/// The server imports ONLY this trait — no model/engine specifics leak.
///
/// Rule: one implementation per engine type (MLX, CPU, Metal, SSD).
/// The dispatch crate selects which one via feature flags.
pub trait InferenceEngine: Send + Sync {
    /// Model identifier (e.g., "qwen3-1.7b-4bit").
    fn model_id(&self) -> &str;

    /// Engine name (e.g., "mlx-compiled", "metal-native", "cpu").
    fn engine_name(&self) -> &str;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// End-of-sequence token IDs.
    fn eos_ids(&self) -> &[u32];

    /// Encode text to token IDs.
    fn encode(&self, text: &str) -> Result<Vec<u32>>;

    /// Encode chat messages (applies chat template).
    fn encode_chat(&self, messages: &[ChatMsg]) -> Result<Vec<u32>>;

    /// Decode token IDs to text.
    fn decode(&self, ids: &[u32]) -> String;

    /// Prefill: process prompt tokens, return logits for last position.
    fn prefill(&self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Step: given previous token, generate next token.
    /// Returns (next_token_id, is_eos).
    fn step(&self, token: u32) -> Result<(u32, bool)>;

    /// Reset engine state (KV cache, positions, etc.).
    fn reset(&self);
}

/// Hardware-agnostic engine factory.
///
/// Creates Model instances from model directories.
pub trait Engine: Send + Sync {
    /// Engine name.
    fn name(&self) -> &str;

    /// Load a model from a directory path.
    fn load_model(&self, path: &Path) -> Result<Box<dyn Model>>;

    /// Whether this engine is available on the current system.
    fn is_available(&self) -> bool;
}

/// Hardware-agnostic model — raw forward pass.
///
/// No tokenization, no sampling. Pure tensor in → tensor out.
pub trait Model: Send {
    /// Model display name.
    fn name(&self) -> &str;

    /// Full forward pass: tokens → logits.
    fn forward(&mut self, tokens: &[u32]) -> Result<Tensor>;

    /// Single decode step: token → next_token.
    fn step(&mut self, token: u32) -> Result<u32>;

    /// Reset model state.
    fn reset(&mut self);

    /// Current offset (number of tokens processed).
    fn offset(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Model configuration.
    fn config(&self) -> &ModelConfig;

    /// Architecture type.
    fn architecture(&self) -> Architecture;
}

/// Backend-specific engine pattern (Metal, MLX, SSD all share this shape).
///
/// Unlike `InferenceEngine`, this has `&mut self` (not shared) and
/// doesn't handle tokenization.
pub trait BackendEngine: Send + Sync {
    /// Load model from directory.
    fn load(&mut self, model_dir: &Path) -> Result<()>;

    /// Forward pass: tokens → logits.
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Single step: previous token → next token.
    fn step(&mut self, token_id: u32) -> Result<u32>;

    /// Prefill: process prompt tokens → logits for last position.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Reset engine state.
    fn reset(&mut self);

    /// Engine name.
    fn name(&self) -> &str;

    /// Expected tokens per second on this hardware.
    fn expected_tps(&self) -> f32;
}

/// Speculative decoding draft model.
///
/// Provides fast but approximate predictions that the target model
/// can verify in parallel (acceptance/rejection).
pub trait SpecModel: Send {
    /// Generate next token (fast, approximate).
    fn step(&mut self, token: u32) -> Result<u32>;

    /// Verify a sequence of draft tokens against target logits.
    /// Returns logits for parallel verification.
    fn forward_verify(&mut self, tokens: &[u32]) -> Result<Tensor>;

    /// Vocabulary size (must match target model).
    fn vocab_size(&self) -> usize;

    /// Reset cache (called on rejection/rollback).
    fn reset_cache(&mut self);
}
