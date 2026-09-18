//! Attention traits — the core contracts for attention implementations.
//!
//! Implementations wire attention + KV cache together per backend:
//! - `nxm-attention-cpu` (candle SDPA + PagedAttention)
//! - `nxm-attention-mlx` (MLX compiled SDPA + R-SWA cache)
//! - `nxm-attention-metal` (FlashAttention Metal + ring buffer cache)

use nxm_cache_core::KvCacheBackend;
use nxm_ops_core::Tensor;

use crate::config::AttentionConfig;
use crate::error::Result;
use crate::mask::Mask;

/// Core attention trait — scaled dot-product attention with KV cache.
///
/// One instance per layer. Manages its own KV cache internally.
pub trait Attention: Send {
    /// Compute attention: O = softmax(Q K^T / scale) V.
    ///
    /// Pure computation, no cache interaction.
    ///
    /// - `q`: [batch, n_heads, seq_len, head_dim]
    /// - `k`: [batch, n_kv_heads, seq_len, head_dim]
    /// - `v`: [batch, n_kv_heads, seq_len, head_dim]
    /// - Returns: [batch, n_heads, seq_len, head_dim]
    fn forward(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Mask>,
        scale: f32,
    ) -> Result<Tensor>;

    /// Compute attention with KV cache (decode path).
    ///
    /// Appends k/v to cache, then attends over full cached K/V.
    /// This is the hot path for token generation.
    ///
    /// - `q`: [batch, n_heads, 1, head_dim] (single query token)
    /// - `k`: [batch, n_kv_heads, 1, head_dim]
    /// - `v`: [batch, n_kv_heads, 1, head_dim]
    /// - `position`: absolute position of this token
    /// - Returns: [batch, n_heads, 1, head_dim]
    fn with_cache(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        position: usize,
        mask: Option<&Mask>,
    ) -> Result<Tensor>;

    /// Get the underlying KV cache (for inspection or management).
    fn cache(&self) -> &dyn KvCacheBackend;

    /// Mutable access to KV cache.
    fn cache_mut(&mut self) -> &mut dyn KvCacheBackend;

    /// Get attention config.
    fn config(&self) -> &AttentionConfig;

    /// Reset attention state (clear KV cache).
    fn reset(&mut self);

    /// Current number of cached tokens.
    fn cached_len(&self) -> usize {
        self.cache().len()
    }
}

/// Flash attention — O(N) memory, tiled computation.
///
/// Extends base Attention with block-based forward that avoids
/// materializing the full [seq, seq] attention matrix.
pub trait FlashAttention: Attention {
    /// Flash attention forward (no cache, full sequence).
    ///
    /// Uses tiled computation: O(N * block_size) memory instead of O(N²).
    /// block_size is implementation-chosen based on hardware.
    fn forward_flash(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Mask>,
    ) -> Result<Tensor>;

    /// Supported head dimensions for this flash attention implementation.
    ///
    /// Metal kernels are compiled with function constants for specific head_dims.
    fn supported_head_dims(&self) -> &[usize];

    /// Whether split-K is available (for long KV sequences in decode).
    fn supports_split_k(&self) -> bool {
        false
    }
}

/// Sliding Window Attention — constant-cost decode via R-SWA.
///
/// Reference tokens (prompt) are always visible.
/// Decode tokens rotate in a fixed-size window.
/// Total attention cost: O(ref_len + window) = constant.
pub trait SlidingWindowAttention: Attention {
    /// Size of the sliding decode window.
    fn window_size(&self) -> usize;

    /// Number of reference (prompt) tokens always visible.
    fn reference_len(&self) -> usize;

    /// Set reference tokens (called after prefill).
    ///
    /// These tokens are pinned and never evicted from cache.
    fn set_reference(&mut self, n_tokens: usize);

    /// Current offset in the sliding window (wraps around).
    fn window_offset(&self) -> usize;
}
