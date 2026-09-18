//! GQA (Grouped-Query Attention) — stub implementation.
//!
//! **NOTE**: Stub for workspace structure.

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::MlxCtx;
use crate::QuantWeights;
use anyhow::Result;

/// Project Q, K, V from input.
pub fn project_qkv(
    _ctx: &MlxCtx,
    _x: mlx_array,
    _q_proj: &QuantWeights,
    _k_proj: &QuantWeights,
    _v_proj: &QuantWeights,
    _nh: i32,
    _nkv: i32,
    _hd: i32,
    _seq_len: i32,
) -> Result<(mlx_array, mlx_array, mlx_array)> {
    anyhow::bail!("project_qkv not implemented - needs mlx-c FFI")
}

/// Optional QK norm.
pub fn optional_qk_norm(
    _ctx: &MlxCtx,
    _q: mlx_array,
    _k: mlx_array,
    _q_norm: Option<mlx_array>,
    _k_norm: Option<mlx_array>,
) -> Result<(mlx_array, mlx_array)> {
    anyhow::bail!("optional_qk_norm not implemented - needs mlx-c FFI")
}

/// Transpose for attention.
pub fn transpose_for_attn(
    _ctx: &MlxCtx,
    _q: mlx_array,
    _k: mlx_array,
    _v: mlx_array,
) -> Result<(mlx_array, mlx_array, mlx_array)> {
    anyhow::bail!("transpose_for_attn not implemented - needs mlx-c FFI")
}

/// KV cache concat.
pub fn kv_cache_concat(
    _ctx: &MlxCtx,
    _ck: Option<mlx_array>,
    _k: mlx_array,
    _cv: Option<mlx_array>,
    _v: mlx_array,
) -> Result<(mlx_array, mlx_array)> {
    anyhow::bail!("kv_cache_concat not implemented - needs mlx-c FFI")
}

/// Attend and project output.
pub fn attend_and_project(
    _ctx: &MlxCtx,
    _q: mlx_array,
    _k: mlx_array,
    _v: mlx_array,
    _o_proj: &QuantWeights,
    _scale: f32,
    _is_prefill: bool,
) -> Result<mlx_array> {
    anyhow::bail!("attend_and_project not implemented - needs mlx-c FFI")
}