//! Attention utilities for ops crate.
//!
//! **NOTE**: Stub for workspace structure.

use crate::ffi::mlx_array;

pub fn gqa_attention(
    _ctx: &crate::MlxCtx,
    _q: mlx_array,
    _k: mlx_array,
    _v: mlx_array,
    _scale: f32,
    _mask: Option<mlx_array>,
) -> mlx_array {
    panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c.")
}

pub fn sliding_window_attention(
    _ctx: &crate::MlxCtx,
    _q: mlx_array,
    _k: mlx_array,
    _v: mlx_array,
    _window: usize,
    _scale: f32,
    _mask: Option<mlx_array>,
) -> mlx_array {
    panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c.")
}
