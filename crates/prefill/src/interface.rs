//! Interface — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use anyhow::Result;

pub struct Interface;

impl Interface {
    pub fn new() -> Result<Self> {
        anyhow::bail!("Interface not implemented - needs mlx-c FFI")
    }
}