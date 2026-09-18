//! Expert store traits for MoE models on limited RAM.
//!
//! Enables 35B-80B MoE models on 16GB machines via:
//! - mmap + madvise for zero-copy expert loading
//! - LRU dequant cache (hot experts stay in RAM)
//! - Predictive prefetch from routing patterns
//!
//! Based on: nexum SSD engine + oMLX tiered cache design.

use std::path::Path;

use crate::error::Result;

/// Expert store — manages MoE expert weights on disk/SSD.
///
/// The model has N experts per MoE layer, but only top-K are active
/// per token. The store pages experts in/out of RAM as needed.
pub trait ExpertStore: Send + Sync {
    /// Prefetch experts into RAM (async, non-blocking).
    ///
    /// Called ahead of time based on routing predictions.
    fn prefetch(&self, layer: usize, experts: &[usize]) -> Result<()>;

    /// Release experts from RAM (hint: no longer needed).
    fn release(&self, layer: usize, experts: &[usize]) -> Result<()>;

    /// Check if a specific layer uses MoE (vs dense).
    fn is_moe(&self, layer: usize) -> bool;

    /// Number of experts in a MoE layer.
    fn expert_count(&self, layer: usize) -> usize;

    /// Hidden size of the model.
    fn hidden_size(&self) -> usize;

    /// Intermediate size per expert.
    fn intermediate_size(&self) -> usize;

    /// Check if an expert is currently hot (in RAM).
    fn is_hot(&self, layer: usize, expert: usize) -> bool;

    /// Current RAM usage for cached experts.
    fn memory_usage(&self) -> usize;

    /// Maximum RAM budget for expert cache.
    fn memory_budget(&self) -> usize;
}

/// Expert page — metadata for a single expert's weight page.
#[derive(Debug, Clone)]
pub struct ExpertPage {
    /// Layer index.
    pub layer: usize,
    /// Expert index within the layer.
    pub expert: usize,
    /// File offset for mmap.
    pub offset: u64,
    /// Size in bytes.
    pub size: usize,
    /// Whether currently resident in RAM.
    pub resident: bool,
    /// Last access timestamp (for LRU eviction).
    pub last_access: u64,
}

/// Configuration for expert store.
#[derive(Debug, Clone)]
pub struct ExpertStoreConfig {
    /// Path to model weights directory.
    pub model_dir: Box<Path>,
    /// Maximum RAM budget for expert cache (bytes).
    pub memory_budget: usize,
    /// Number of experts to prefetch ahead.
    pub prefetch_count: usize,
    /// Whether to use mmap (vs read into buffer).
    pub use_mmap: bool,
    /// Whether to use MADV_DONTNEED for release (vs munmap).
    pub use_madvise: bool,
}
