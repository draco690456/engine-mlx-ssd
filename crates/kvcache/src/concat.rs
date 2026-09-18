//! Concat KV cache — raw concatenation, no compression.
//!
//! **NOTE**: Stub implementation. Full MLX-C FFI needed.

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;
use crate::KvCache;
use anyhow::Result;

/// Raw concatenation cache. Stores K/V as mlx_array on the compute graph.
pub struct ConcatCache {
    k: Option<mlx_array>,
    v: Option<mlx_array>,
    seq_len: usize,
}

unsafe impl Send for ConcatCache {}
unsafe impl Sync for ConcatCache {}

impl ConcatCache {
    pub fn new() -> Self {
        Self { k: None, v: None, seq_len: 0 }
    }
}

impl Default for ConcatCache {
    fn default() -> Self { Self::new() }
}

impl KvCache for ConcatCache {
    fn append(&mut self, ctx: &MlxCtx, k: mlx_array, v: mlx_array) -> Result<()> {
        anyhow::bail!("ConcatCache::append not implemented - needs mlx-c FFI")
    }
    fn get(&self, _ctx: &MlxCtx) -> Result<(mlx_array, mlx_array)> {
        anyhow::bail!("ConcatCache::get not implemented - needs mlx-c FFI")
    }
    fn len(&self) -> usize { self.seq_len }
    fn reset(&mut self) { self.k = None; self.v = None; self.seq_len = 0; }
    fn memory_bytes(&self) -> usize { 0 }
}