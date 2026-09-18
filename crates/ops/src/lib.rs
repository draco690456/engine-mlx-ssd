//! # engine-mlx-ops
//!
//! Atomic MLX-C FFI operations for Nexum model crates.
//!
//! **NOTE**: Stub implementation for workspace structure.
//! Full implementation requires MLX-C headers.

pub mod attention;
pub mod compile;
pub mod ctx;
pub mod ffi;
pub mod kv_traits;
pub mod linear;
pub mod quant_cache;

pub use engine_mlx_ffi::*;
pub use kv_traits::{KvCacheBackend, TensorRef};
