//! Rotating KV cache — sliding window with fixed memory.
//!
//! **NOTE**: Stub implementation. Full MLX-C FFI needed.

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;
use crate::KvCache;
use anyhow::Result;

/// Rotating (sliding window) KV cache.
pub struct RotatingCache {
    k: Option<mlx_array>,
    v: Option<mlx_array>,
    seq_len: usize,
}

unsafe impl Send for RotatingCache {}
unsafe impl Sync for RotatingCache {}

impl RotatingCache {
    pub fn new() -> Self {
        Self { k: None, v: None, seq_len: 0 }
    }
}

impl KvCache for RotatingCache {
    fn append(&mut self, _ctx: &MlxCtx, _k: mlx_array, _v: mlx_array) -> Result<()> {
        anyhow::bail!("RotatingCache::append not implemented - needs mlx-c FFI")
    }
    fn get(&self, _ctx: &MlxCtx) -> Result<(mlx_array, mlx_array)> {
        anyhow::bail!("RotatingCache::get not implemented - needs mlx-c FFI")
    }
    fn len(&self) -> usize { self.seq_len }
    fn reset(&mut self) { self.k = None; self.v = None; self.seq_len = 0; }
    fn memory_bytes(&self) -> usize { 0 }
}