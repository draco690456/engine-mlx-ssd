//! Error types for cache operations.

use thiserror::Error;

/// Errors from KV cache operations.
#[derive(Error, Debug)]
pub enum CacheError {
    /// Shape mismatch when storing/loading.
    #[error("cache shape error: {0}")]
    Shape(String),

    /// Configuration error.
    #[error("cache config error: {0}")]
    Config(String),

    /// Runtime error (OOM, corruption, etc.).
    #[error("cache runtime error: {0}")]
    Runtime(String),

    /// I/O error (disk cache, mmap).
    #[error("cache I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Operation not supported by this backend.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Expert not found or not resident.
    #[error("expert error: layer={layer}, expert={expert}: {msg}")]
    Expert {
        layer: usize,
        expert: usize,
        msg: String,
    },
}

/// Result type for cache operations.
pub type Result<T> = std::result::Result<T, CacheError>;
