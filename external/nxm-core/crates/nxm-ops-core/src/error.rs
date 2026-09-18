//! Error types for operations.

use thiserror::Error;

use crate::tensor::{DType, Shape};

/// Errors from tensor operations.
#[derive(Error, Debug)]
pub enum OpsError {
    /// Shape mismatch between operands.
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: Shape, actual: Shape },

    /// Incompatible dimensions for operation.
    #[error("incompatible dimensions: {0}")]
    IncompatibleDims(String),

    /// Unsupported data type for this operation.
    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(DType),

    /// Invalid quantization parameters.
    #[error("quantization error: {0}")]
    Quantization(String),

    /// Numeric error (overflow, NaN, etc.).
    #[error("numeric error: {0}")]
    Numeric(String),

    /// Backend-specific error.
    #[error("backend error: {0}")]
    Backend(String),
}

/// Result type for operations.
pub type Result<T> = std::result::Result<T, OpsError>;
