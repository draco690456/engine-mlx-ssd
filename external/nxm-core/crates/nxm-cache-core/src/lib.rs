//! # nxm-cache-core
//!
//! KV cache and expert store trait contracts for Nexum inference.
//!
//! This crate defines:
//! - `KvCacheBackend` — store/load/append KV tensors (with compression support)
//! - `CacheResolver` — per-layer quantization strategy selection
//! - `KvScoring` — importance scoring for cache eviction (EPiCache)
//! - `ExpertStore` — MoE expert paging for large models on limited RAM
//! - Types: `TensorRef`, `CompressedTensor`, `QuantType`, `CacheConfig`, `BlockAddr`
//!
//! Implementations live in Layer 2:
//! - `nxm-cache-memory` (RAM concatenate / ring buffer)
//! - `nxm-cache-ssd` (tiered hot/cold with SSD persistence)
//! - `nxm-cache-turboquant` (6.4× FWHT + Lloyd-Max compression)
//! - `nxm-cache-rotor/iso/planar/paro` (4× rotation-based compression)

pub mod error;
pub mod expert;
pub mod traits;
pub mod types;

pub use error::{CacheError, Result};
pub use expert::{ExpertPage, ExpertStore, ExpertStoreConfig};
pub use traits::{CacheResolver, KvCacheBackend, KvScoring};
pub use types::{AttendConfig, BlockAddr, CacheConfig, CompressedTensor, QuantType, TensorRef};
