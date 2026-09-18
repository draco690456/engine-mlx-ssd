//! Server — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use anyhow::Result;

pub struct Server;

impl Server {
    pub fn new() -> Result<Self> {
        anyhow::bail!("Server not implemented - needs mlx-c FFI")
    }
}