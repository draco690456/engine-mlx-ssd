//! Engine — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use anyhow::Result;

pub struct Engine;

impl Engine {
    pub fn new() -> Result<Self> {
        anyhow::bail!("Engine not implemented - needs mlx-c FFI")
    }
}