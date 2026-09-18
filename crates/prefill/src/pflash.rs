//! PFlash — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use engine_mlx_ops::MlxCtx;
use anyhow::Result;

pub struct PFlash;

impl PFlash {
    pub fn new(_ctx: &MlxCtx) -> Result<Self> {
        anyhow::bail!("PFlash not implemented - needs mlx-c FFI")
    }
}