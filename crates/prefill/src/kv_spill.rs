//! KV Spill — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use engine_mlx_ops::MlxCtx;
use anyhow::Result;

pub struct KvSpill;

impl KvSpill {
    pub fn new(_ctx: &MlxCtx) -> Result<Self> {
        anyhow::bail!("KvSpill not implemented - needs mlx-c FFI")
    }
}