//! Gated attention — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;
use crate::QuantWeights;
use anyhow::Result;

/// Project Q with gating.
pub fn project_q_gated(
    _ctx: &MlxCtx,
    _x: mlx_array,
    _q_proj: &QuantWeights,
    _gate_proj: &QuantWeights,
    _nh: i32,
    _hd: i32,
    _seq_len: i32,
) -> Result<(mlx_array, mlx_array)> {
    anyhow::bail!("project_q_gated not implemented - needs mlx-c FFI")
}

/// Apply output gate.
pub fn apply_output_gate(
    _ctx: &MlxCtx,
    _attn_out: mlx_array,
    _gate: mlx_array,
) -> Result<mlx_array> {
    anyhow::bail!("apply_output_gate not implemented - needs mlx-c FFI")
}