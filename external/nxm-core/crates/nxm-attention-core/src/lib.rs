//! # nxm-attention-core
//!
//! Attention mechanism trait contracts for Nexum inference.
//!
//! This crate defines:
//! - `Attention` trait — base contract for attention computation
//! - `FlashAttention` trait — O(n) memory flash attention
//! - `SlidingWindowAttention` trait — R-SWA constant-cost decode
//! - `Mask` enum — causal, sliding, custom mask types
//! - `AttentionConfig` — head count, GQA/MQA/MHA/MLA/GDN config
//!
//! Key design: attention and KV cache are tightly coupled in implementations
//! (zero-copy, same memory). The traits define the CONTRACT; implementations
//! in Layer 2 (`nxm-attention-cpu`, `nxm-attention-mlx`, `nxm-attention-metal`)
//! wire attention + cache together for maximum performance.

pub mod config;
pub mod error;
pub mod mask;
pub mod traits;

pub use config::{AttentionConfig, AttentionType};
pub use error::{AttentionError, Result};
pub use mask::Mask;
pub use traits::{Attention, FlashAttention, SlidingWindowAttention};
