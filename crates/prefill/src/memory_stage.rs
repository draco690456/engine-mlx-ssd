//! Memory Stage — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use engine_mlx_ops::MlxCtx;
use anyhow::Result;

pub struct MemoryStage;

impl MemoryStage {
    pub fn new(_ctx: &MlxCtx) -> Result<Self> {
        anyhow::bail!("MemoryStage not implemented - needs mlx-c FFI")
    }
}