//! Pipeline — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use engine_mlx_ops::MlxCtx;
use anyhow::Result;

pub struct Pipeline;

impl Pipeline {
    pub fn new(_ctx: &MlxCtx) -> Result<Self> {
        anyhow::bail!("Pipeline not implemented - needs mlx-c FFI")
    }
}