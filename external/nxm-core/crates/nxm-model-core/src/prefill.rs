//! Prefill pipeline trait.
//!
//! The prefill pipeline orchestrates how prompts are processed before
//! decode begins. It may include: memory injection, prefix cache lookup,
//! PFlash compression, chunked batch prefill, cache store.

use crate::error::Result;

/// Result of the prefill pipeline.
#[derive(Debug, Clone)]
pub struct PrefillResult {
    /// Logits from the last position.
    pub logits: Vec<f32>,
    /// Total tokens processed (may be less than input if compressed).
    pub tokens_processed: usize,
    /// Whether prefix cache was hit.
    pub prefix_cache_hit: bool,
    /// Number of tokens that were compressed/skipped.
    pub tokens_compressed: usize,
}

/// Prefill pipeline — orchestrates prompt processing.
///
/// Stages (optional, implementation-dependent):
/// 1. Memory injection (prepend relevant memories)
/// 2. Prefix cache lookup (skip already-processed prefix)
/// 3. PFlash compression (score + select important tokens for long prompts)
/// 4. Chunked batch prefill (process in chunks to avoid OOM)
/// 5. Cache store (persist KV for future prefix sharing)
pub trait PrefillPipeline: Send {
    /// Run the prefill pipeline.
    ///
    /// `tokens`: full prompt token IDs
    /// `has_tools`: whether tool definitions are present (affects scoring)
    /// `forward_fn`: forward pass function (engine-specific)
    fn run(
        &mut self,
        tokens: &[u32],
        has_tools: bool,
        forward_fn: &mut dyn FnMut(&[u32]) -> Result<Vec<f32>>,
    ) -> Result<PrefillResult>;

    /// Simplified run without tool awareness.
    fn run_simple(
        &mut self,
        tokens: &[u32],
        forward_fn: &mut dyn FnMut(&[u32]) -> Result<Vec<f32>>,
    ) -> Result<PrefillResult> {
        self.run(tokens, false, forward_fn)
    }

    /// Reset pipeline state.
    fn reset(&mut self);
}
