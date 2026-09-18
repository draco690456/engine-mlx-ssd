//! Error types for attention operations.

use thiserror::Error;

/// Errors from attention computation.
#[derive(Error, Debug)]
pub enum AttentionError {
    /// Head dimension not supported by this kernel.
    #[error("unsupported head dim: {0} (supported: {1:?})")]
    UnsupportedHeadDim(usize, Vec<usize>),

    /// Shape mismatch in attention tensors.
    #[error("attention shape error: {0}")]
    Shape(String),

    /// Cache error propagated.
    #[error("cache error: {0}")]
    Cache(#[from] nxm_cache_core::CacheError),

    /// Operations error propagated.
    #[error("ops error: {0}")]
    Ops(#[from] nxm_ops_core::OpsError),

    /// Configuration error.
    #[error("attention config error: {0}")]
    Config(String),

    /// Numeric error (NaN, overflow in softmax, etc.).
    #[error("attention numeric error: {0}")]
    Numeric(String),
}

/// Result type for attention operations.
pub type Result<T> = std::result::Result<T, AttentionError>;
