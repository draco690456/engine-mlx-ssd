//! Loader — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use anyhow::Result;

pub struct Loader;

impl Loader {
    pub fn new() -> Result<Self> {
        anyhow::bail!("Loader not implemented - needs mlx-c FFI")
    }
}