//! FP8 E4M3FN KV cache — simulated round-trip quantization.
//!
//! **NOTE**: Stub implementation. Full MLX-C FFI needed.

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;
use crate::KvCache;
use anyhow::Result;

/// FP8 E4M3FN cache with simulated round-trip.
pub struct Fp8Cache {
    k: Option<mlx_array>,
    v: Option<mlx_array>,
    seq_len: usize,
}

unsafe impl Send for Fp8Cache {}
unsafe impl Sync for Fp8Cache {}

impl Fp8Cache {
    pub fn new() -> Self {
        Self { k: None, v: None, seq_len: 0 }
    }
}

impl KvCache for Fp8Cache {
    fn append(&mut self, _ctx: &MlxCtx, _k: mlx_array, _v: mlx_array) -> Result<()> {
        anyhow::bail!("Fp8Cache::append not implemented - needs mlx-c FFI")
    }
    fn get(&self, _ctx: &MlxCtx) -> Result<(mlx_array, mlx_array)> {
        anyhow::bail!("Fp8Cache::get not implemented - needs mlx-c FFI")
    }
    fn len(&self) -> usize { self.seq_len }
    fn reset(&mut self) { self.k = None; self.v = None; self.seq_len = 0; }
    fn memory_bytes(&self) -> usize { 0 }
}