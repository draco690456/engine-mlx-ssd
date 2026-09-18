//! Linear operations.
//!
//! **NOTE**: Stub for workspace structure.

use crate::ffi::mlx_array;
use crate::MlxCtx;

pub fn linear(
    _ctx: &MlxCtx,
    _x: mlx_array,
    _weight: mlx_array,
    _bias: Option<mlx_array>,
) -> mlx_array {
    panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c.")
}

pub fn linear_quantized(
    _ctx: &MlxCtx,
    _x: mlx_array,
    _weight: mlx_array,
    _scales: mlx_array,
    _biases: mlx_array,
    _group_size: i32,
    _bits: i32,
) -> mlx_array {
    panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c.")
}
