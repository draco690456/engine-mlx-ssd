//! # engine-mlx-attention
//!
//! Complete attention layers for MLX-C — GQA, sliding window, gated, GDN.
//!
//! **NOTE**: This is a minimal stub for workspace structure.
//! Full MLX-C integration requires complete MLX-C FFI implementation.

pub mod gated;
pub mod gdn;
pub mod gqa;
pub mod sliding_window;

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;
use anyhow::Result;

/// Placeholder QuantWeights for attention ops.
pub struct QuantWeights {
    pub weight: mlx_array,
    pub scales: mlx_array,
    pub biases: mlx_array,
    pub group_size: i32,
    pub bits: i32,
    pub mode: String,
}

/// Placeholder GdnConfig.
pub struct GdnConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

/// Placeholder GdnState.
pub struct GdnState {
    pub h: Option<mlx_array>,
}

impl GdnConfig {
    pub fn new(hidden_dim: usize, num_heads: usize, head_dim: usize) -> Self {
        Self { hidden_dim, num_heads, head_dim }
    }
}

// Re-export from ops crate for convenience
pub use engine_mlx_ops::KvCacheBackend;
pub use engine_mlx_ops::TensorRef;