//! Prefix Cache — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use engine_mlx_ops::MlxCtx;
use anyhow::Result;

pub struct PrefixCache;

impl PrefixCache {
    pub fn new(_ctx: &MlxCtx) -> Result<Self> {
        anyhow::bail!("PrefixCache not implemented - needs mlx-c FFI")
    }
}