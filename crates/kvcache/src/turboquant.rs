//! TurboQuant KV cache — Lloyd-Max codebook quantization.
//!
//! **NOTE**: Stub implementation. Full MLX-C FFI needed.

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;
use crate::KvCache;
use anyhow::Result;

/// TurboQuant configuration.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    pub bits: u8,
    pub group_size: usize,
}

impl Default for TurboQuantConfig {
    fn default() -> Self { Self { bits: 4, group_size: 64 } }
}

/// TurboQuant KV cache.
pub struct TurboQuantCache {
    k: Option<mlx_array>,
    v: Option<mlx_array>,
    seq_len: usize,
}

unsafe impl Send for TurboQuantCache {}
unsafe impl Sync for TurboQuantCache {}

impl TurboQuantCache {
    pub fn new() -> Self {
        Self { k: None, v: None, seq_len: 0 }
    }
}

impl KvCache for TurboQuantCache {
    fn append(&mut self, _ctx: &MlxCtx, _k: mlx_array, _v: mlx_array) -> Result<()> {
        anyhow::bail!("TurboQuantCache::append not implemented - needs mlx-c FFI")
    }
    fn get(&self, _ctx: &MlxCtx) -> Result<(mlx_array, mlx_array)> {
        anyhow::bail!("TurboQuantCache::get not implemented - needs mlx-c FFI")
    }
    fn len(&self) -> usize { self.seq_len }
    fn reset(&mut self) { self.k = None; self.v = None; self.seq_len = 0; }
    fn memory_bytes(&self) -> usize { 0 }
}