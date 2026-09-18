//! GDN (Gated Delta Network) — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;
use crate::{GdnConfig, GdnState};
use anyhow::Result;

/// Compute gate for GDN.
pub fn compute_gate(
    _ctx: &MlxCtx,
    _x: mlx_array,
    _config: &GdnConfig,
) -> Result<mlx_array> {
    anyhow::bail!("compute_gate not implemented - needs mlx-c FFI")
}

/// GDN step.
pub fn gdn_step(
    _ctx: &MlxCtx,
    _state: &mut GdnState,
    _x: mlx_array,
    _config: &GdnConfig,
) -> Result<mlx_array> {
    anyhow::bail!("gdn_step not implemented - needs mlx-c FFI")
}

/// GDN forward.
pub fn gdn_forward(
    _ctx: &MlxCtx,
    _x: mlx_array,
    _config: &GdnConfig,
) -> Result<mlx_array> {
    anyhow::bail!("gdn_forward not implemented - needs mlx-c FFI")
}