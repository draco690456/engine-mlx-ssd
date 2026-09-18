//! # engine-mlx-prefill
//!
//! Chunked/paged prefill pipeline for MLX-C.
//!
//! **NOTE**: This is a minimal stub for workspace structure.
//! Full MLX-C integration requires complete MLX-C FFI implementation.

pub mod chunked_prefill;
pub mod config;
pub mod interface;
pub mod kv_spill;
pub mod memory_stage;
pub mod pflash;
pub mod pipeline;
pub mod prefix_cache;

use engine_mlx_ops::MlxCtx;
use anyhow::Result;

/// Placeholder for prefill pipeline.
pub struct PrefillPipeline;

impl PrefillPipeline {
    pub fn new(_ctx: &MlxCtx) -> Result<Self> {
        anyhow::bail!("PrefillPipeline not implemented - needs mlx-c FFI")
    }
}