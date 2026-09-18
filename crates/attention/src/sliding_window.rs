//! Sliding Window Attention — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;
use crate::QuantWeights;
use anyhow::Result;

/// Sliding window attention forward.
pub fn sliding_window_attention(
    _ctx: &MlxCtx,
    _q: mlx_array,
    _k: mlx_array,
    _v: mlx_array,
    _window_size: i32,
    _scale: f32,
) -> Result<mlx_array> {
    anyhow::bail!("sliding_window_attention not implemented - needs mlx-c FFI")
}