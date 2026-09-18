//! KV cache backend trait — the core contract for all cache implementations.
//!
//! Implementors:
//! - `nxm-cache-memory` (RAM-only, concatenate or ring buffer)
//! - `nxm-cache-ssd` (tiered hot/cold with SSD persistence)
//! - `nxm-cache-turboquant` (6.4× compressed via Lloyd-Max + FWHT)
//! - `nxm-cache-rotor/iso/planar/paro` (4× compressed variants)

use std::any::Any;

use crate::error::Result;
use crate::types::{AttendConfig, TensorRef};

/// KV cache backend — stores and retrieves key/value tensors.
///
/// Each layer has its own cache instance. The cache may compress,
/// quantize, page to SSD, or simply concatenate — that's the
/// implementor's choice.
pub trait KvCacheBackend: Send + Sync {
    /// Store key/value tensors (overwrite existing).
    fn store(&mut self, keys: TensorRef, values: TensorRef) -> Result<()>;

    /// Append new key/value to existing cache (default: reload + concat + re-store).
    fn append(&mut self, new_keys: TensorRef, new_values: TensorRef) -> Result<()> {
        let (existing_k, existing_v) = self.load()?;
        let merged_k = concat_tensor_ref(&existing_k, &new_keys);
        let merged_v = concat_tensor_ref(&existing_v, &new_values);
        self.store(merged_k, merged_v)
    }

    /// Load full key/value tensors from cache.
    fn load(&self) -> Result<(TensorRef, TensorRef)>;

    /// Number of cached tokens.
    fn len(&self) -> usize;

    /// Whether cache is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reset cache (drop all cached tokens).
    fn reset(&mut self);

    /// Current memory usage in bytes.
    fn memory_usage(&self) -> usize;

    /// Decode cache to f32 (for inspection/debugging).
    /// Returns (keys_f32, values_f32).
    fn decode_to_f32(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        Err(crate::error::CacheError::Unsupported(
            "decode_to_f32 not supported by this backend".into(),
        ))
    }

    /// Compressed attention: compute attention directly on compressed KV.
    ///
    /// If supported, avoids full decompression. Returns None if not supported
    /// (caller falls back to load() + standard attention).
    fn attend_compressed(
        &self,
        query: &[f32],
        config: &AttendConfig,
    ) -> Result<Option<Vec<f32>>> {
        let _ = (query, config);
        Ok(None)
    }

    /// Downcast to concrete type for backend-specific operations.
    fn as_any(&self) -> &dyn Any;
}

/// Cache resolver — selects quantization strategy per layer/head.
///
/// Different layers benefit from different compression levels:
/// - Early layers: more sensitive → less compression
/// - Deep layers: more redundant → aggressive compression
pub trait CacheResolver: Send + Sync {
    /// Resolve the quantization type for a given layer and head.
    fn resolve(&self, layer: usize, head: usize) -> crate::types::QuantType;

    /// Resolve budget (max tokens) for a given layer.
    fn budget(&self, layer: usize) -> usize;
}

/// KV importance scoring for eviction decisions.
///
/// Based on EPiCache (Apple Research): per-head importance scoring
/// with non-uniform budget allocation.
pub trait KvScoring: Send + Sync {
    /// Score importance of each cached token.
    ///
    /// `keys`: current cache keys [n_tokens, head_dim]
    /// `query`: current query [head_dim]
    ///
    /// Returns importance score per token (higher = more important).
    fn score(&self, keys: &TensorRef, query: &[f32]) -> Result<Vec<f32>>;

    /// Select tokens to evict given a budget.
    ///
    /// Returns indices of tokens to KEEP (sorted by position).
    fn select_keep(&self, scores: &[f32], budget: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep: Vec<usize> = indexed.iter().take(budget).map(|(i, _)| *i).collect();
        keep.sort();
        keep
    }
}

/// Concatenate two TensorRef values (f32 path only, fallback).
fn concat_tensor_ref(a: &TensorRef, b: &TensorRef) -> TensorRef {
    match (a, b) {
        (TensorRef::F32(va), TensorRef::F32(vb)) => {
            let mut merged = va.clone();
            merged.extend_from_slice(vb);
            TensorRef::F32(merged)
        }
        _ => {
            // For compressed types, this is a simplistic fallback.
            // Real implementations handle this internally.
            TensorRef::F32(vec![])
        }
    }
}
