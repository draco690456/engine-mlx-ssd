//! Chunked prefill — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use engine_mlx_ops::MlxCtx;
use anyhow::Result;

pub fn chunked_prefill(_ctx: &MlxCtx) -> Result<()> {
    anyhow::bail!("chunked_prefill not implemented - needs mlx-c FFI")
}