//! # nxm-ops-core
//!
//! Core tensor types and operations backend trait for Nexum inference.
//!
//! This crate defines:
//! - `Tensor`, `DType`, `Shape` — CPU-backed value types for trait boundaries
//! - `QuantWeights`, `QuantMode` — quantized weight representation
//! - `OpsBackend` trait — the compute contract all backends implement
//!
//! No implementations here. See:
//! - `nxm-ops-cpu` for candle/BLAS backend
//! - `nxm-ops-mlx` for Apple MLX backend
//! - `nxm-ops-metal` for Metal compute shader backend

pub mod error;
pub mod quant;
pub mod tensor;
pub mod traits;

pub use error::{OpsError, Result};
pub use quant::{QuantMode, QuantWeights};
pub use tensor::{DType, Shape, Tensor};
pub use traits::{MoERoutingOutput, OpsBackend};
