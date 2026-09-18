//! Config — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use anyhow::Result;

pub struct Config;

impl Config {
    pub fn new() -> Result<Self> {
        anyhow::bail!("Config not implemented - needs mlx-c FFI")
    }
}