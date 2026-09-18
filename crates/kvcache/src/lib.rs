//! # engine-mlx-kvcache
//!
//! KV cache backends for MLX-C — concat, rotating, fp8, turboquant.
//!
//! **NOTE**: This is a minimal stub for workspace structure.
//! Full MLX-C integration requires complete MLX-C FFI implementation.

pub mod concat;
pub mod fp8;
pub mod rotating;
pub mod turboquant;

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;
use anyhow::Result;

/// KV cache backend trait.
///
/// All implementations store K/V tensors and provide retrieval for SDPA.
/// Tensors are `mlx_array` handles — they live on the MLX compute graph.
pub trait KvCache: Send + Sync {
    /// Append new K and V for the current step.
    fn append(&mut self, ctx: &MlxCtx, k: mlx_array, v: mlx_array) -> Result<()>;

    /// Get the full accumulated K and V for SDPA.
    fn get(&self, ctx: &MlxCtx) -> Result<(mlx_array, mlx_array)>;

    /// Current sequence length stored in cache.
    fn len(&self) -> usize;

    /// Whether the cache is empty.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Reset the cache (start new generation).
    fn reset(&mut self);

    /// Approximate memory usage in bytes.
    fn memory_bytes(&self) -> usize;
}

// Re-export all cache types
pub use concat::ConcatCache;
pub use fp8::Fp8Cache;
pub use rotating::RotatingCache;
pub use turboquant::{TurboQuantConfig, TurboQuantCache};