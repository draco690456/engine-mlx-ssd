//! # nxm-model-core
//!
//! Model, engine, and architecture trait contracts for Nexum inference.
//!
//! This crate defines:
//! - `InferenceEngine` — the server-facing contract (THE boundary)
//! - `Engine` — hardware-agnostic engine factory
//! - `Model` — hardware-agnostic model forward pass
//! - `BackendEngine` — common pattern for backend-specific engines
//! - `SpecModel` — speculative decoding draft/verify model
//! - `Sampler` — token sampling trait
//! - `PrefillPipeline` — prefill orchestration trait
//! - `Architecture` — enum of all supported model architectures
//! - `ModelConfig` — model hyperparameters
//! - Types: `ChatMsg`, `ChatMessage`, `Role`, etc.
//!
//! Architectural rule: the server ONLY sees `InferenceEngine`. Everything
//! below is implementation detail hidden behind dispatch.

pub mod architecture;
pub mod config;
pub mod error;
pub mod prefill;
pub mod sampling;
pub mod traits;
pub mod types;

pub use architecture::Architecture;
pub use config::{EngineKind, InferenceConfig, ModelConfig};
pub use error::{EngineError, ModelError, Result};
pub use prefill::{PrefillPipeline, PrefillResult};
pub use sampling::{Sampler, SamplingConfig};
pub use traits::{BackendEngine, Engine, InferenceEngine, Model, SpecModel};
pub use types::{ChatMessage, ChatMsg, Role};
