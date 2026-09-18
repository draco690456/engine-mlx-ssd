//! Quantized KV cache.
//!
//! **NOTE**: Stub for workspace structure.

use crate::ffi::mlx_array;

pub struct QuantCache {
    _private: (),
}

impl QuantCache {
    pub fn new(_capacity: usize, _head_dim: usize, _dtype: u32) -> Self {
        panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c.")
    }

    pub fn append(&mut self, _k: mlx_array, _v: mlx_array) {
        panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c.")
    }

    pub fn get(&self, _start: usize, _len: usize) -> (mlx_array, mlx_array) {
        panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c.")
    }
}
