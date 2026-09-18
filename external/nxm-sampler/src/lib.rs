//! # nxm_sampler
//!
//! GPU-resident token sampler for Apple Silicon Metal.
//!
//! The critical insight: at 400+ tok/s decode, reading logits back to CPU
//! for sampling costs 0.1-0.5ms per token — that's 5-20% of the decode budget.
//! By keeping logits on GPU and sampling there, we eliminate this entirely.
//!
//! ## Supported strategies
//!
//! | Strategy | Kernel | Randomness |
//! |----------|--------|------------|
//! | Greedy | `argmax_sample` | None — always pick highest |
//! | Top-K | `top_k_filter` + categorical | Keep only K highest |
//! | Top-P (nucleus) | `top_p_filter` + categorical | Keep until cumprob ≥ P |
//! | Temperature | `softmax_temperature` | Scale before softmax |
//! | Repetition penalty | `repetition_penalty` | Penalize seen tokens |
//!
//! ## Usage
//!
//! ```rust,ignore
//! let sampler = GpuSampler::new(&device, SamplerConfig {
//!     temperature: 0.7,
//!     top_k: 40,
//!     top_p: 0.9,
//!     repetition_penalty: 1.1,
//!     strategy: SamplerStrategy::TopK,
//! })?;
//!
//! // In the decode loop (logits stay on GPU!):
//! let token_id = sampler.sample(&encoder, &logits_buffer)?;
//! // token_id is a single u32 read from a 4-byte GPU buffer — NOT the full vocab
//! ```
//!
//! ## Architecture
//!
//! ```text
//! logits [vocab_size] on GPU
//!     │
//!     ├─ repetition_penalty (in-place, penalize context tokens)
//!     │
//!     ├─ top_k_filter (zero out below k-th largest)
//!     │   OR top_p_filter (zero out below cumulative P)
//!     │
//!     ├─ softmax_temperature (temperature + softmax → probs)
//!     │
//!     └─ argmax_sample (greedy) OR categorical_sample (random)
//!           │
//!           ▼
//!     result_token [1] on GPU  ← read back ONLY this (4 bytes, not vocab×4)
//! ```

#![cfg(target_os = "macos")]

pub mod config;
pub mod dispatch;

pub use config::{SamplerConfig, SamplerStrategy};
pub use dispatch::GpuSampler;
