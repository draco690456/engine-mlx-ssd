//! Operations backend trait — the core contract for all compute backends.
//!
//! Implementors:
//! - `nxm-ops-cpu` (candle/BLAS/rayon)
//! - `nxm-ops-mlx` (Apple MLX via mlx-c FFI)
//! - `nxm-ops-metal` (Apple Metal compute shaders)
//!
//! Design: stateless operations. Backend holds device context (GPU handle,
//! thread pool, etc.) but operations are pure functions on tensors.

use crate::error::Result;
use crate::quant::QuantWeights;
use crate::tensor::Tensor;

/// Routing output from MoE gating.
#[derive(Debug, Clone)]
pub struct MoERoutingOutput {
    /// Selected expert indices per token: [n_tokens, top_k].
    pub indices: Vec<u32>,
    /// Routing weights per token: [n_tokens, top_k] (softmax-normalized).
    pub weights: Vec<f32>,
    /// Number of tokens.
    pub n_tokens: usize,
    /// Top-K experts selected.
    pub top_k: usize,
    /// Auxiliary load-balancing loss (for training; 0.0 in inference).
    pub aux_loss: f32,
}

/// Operations backend — all compute ops needed for transformer inference.
///
/// All methods take immutable `&self` — the backend is shared across layers.
/// Tensor allocation is backend-internal (may use GPU memory pools).
pub trait OpsBackend: Send + Sync {
    // ── Normalization ──

    /// RMS normalization: y = x / rms(x) * weight.
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;

    /// Layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias.
    fn layer_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor>;

    // ── Linear algebra ──

    /// Dense matrix multiply: C = A @ B.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Quantized matrix multiply: y = x @ dequant(weights).
    fn qmatmul(&self, x: &Tensor, weights: &QuantWeights) -> Result<Tensor>;

    // ── Positional encoding ──

    /// Rotary position embedding (RoPE).
    /// Returns (q_rotated, k_rotated).
    fn rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)>;

    // ── Activations ──

    /// SiLU activation: x * sigmoid(x).
    fn silu(&self, x: &Tensor) -> Result<Tensor>;

    /// GELU activation.
    fn gelu(&self, x: &Tensor) -> Result<Tensor>;

    /// Fused SwiGLU: silu(gate) * up.
    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor>;

    /// Softmax along given dimension.
    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor>;

    // ── Elementwise ──

    /// Element-wise addition: a + b.
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Element-wise multiplication: a * b.
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Scalar multiply: a * scalar.
    fn scale(&self, a: &Tensor, scalar: f32) -> Result<Tensor>;

    // ── MoE routing ──

    /// MoE top-k expert routing: select top_k experts per token.
    ///
    /// `hidden`: [n_tokens, hidden_size]
    /// `gate`: [hidden_size, n_experts] gate weight matrix
    fn moe_route(
        &self,
        hidden: &Tensor,
        gate: &Tensor,
        top_k: usize,
    ) -> Result<MoERoutingOutput>;

    // ── Embedding ──

    /// Token embedding lookup.
    ///
    /// `token_ids`: [n_tokens] u32
    /// `embed_table`: [vocab_size, hidden_dim]
    /// Returns: [n_tokens, hidden_dim]
    fn embed(&self, token_ids: &Tensor, embed_table: &Tensor) -> Result<Tensor>;
}
