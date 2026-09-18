//! Error types for model and engine operations.

use thiserror::Error;

/// Errors from model operations.
#[derive(Error, Debug)]
pub enum ModelError {
    /// Weight not found in model file.
    #[error("weight not found: {0}")]
    WeightNotFound(String),

    /// Shape mismatch when loading weights.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Model file format error.
    #[error("format error: {0}")]
    Format(String),

    /// Architecture not supported.
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    /// Forward pass error.
    #[error("forward error: {0}")]
    Forward(String),
}

/// Errors from engine operations.
#[derive(Error, Debug)]
pub enum EngineError {
    /// Model loading failed.
    #[error("load error: {0}")]
    Load(String),

    /// Inference failed.
    #[error("inference error: {0}")]
    Inference(String),

    /// Configuration error.
    #[error("config error: {0}")]
    Config(String),

    /// Engine not available on this platform.
    #[error("engine not available: {0}")]
    NotAvailable(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Model error.
    #[error("model error: {0}")]
    Model(#[from] ModelError),

    /// Ops error.
    #[error("ops error: {0}")]
    Ops(#[from] nxm_ops_core::OpsError),

    /// Cache error.
    #[error("cache error: {0}")]
    Cache(#[from] nxm_cache_core::CacheError),

    /// Attention error.
    #[error("attention error: {0}")]
    Attention(#[from] nxm_attention_core::AttentionError),
}

/// Result type for engine/model operations.
pub type Result<T> = std::result::Result<T, EngineError>;
