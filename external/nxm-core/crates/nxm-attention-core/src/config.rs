//! Attention configuration types.

/// Type of attention mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionType {
    /// Multi-Head Attention (all heads are unique).
    MHA,
    /// Grouped Query Attention (N heads share K/V groups).
    GQA,
    /// Multi-Query Attention (all heads share 1 K/V).
    MQA,
    /// Multi-Head Latent Attention (DeepSeek-V2: latent KV compression).
    MLA,
    /// Gated Delta Network (linear recurrence, O(1) memory per step).
    GatedDeltaNet,
    /// Parallax: local softmax window + linear global correction.
    Parallax,
    /// Sigmoid per-head gating (Laguna XS.2 style).
    SigmoidGated,
}

/// Configuration for an attention layer.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (GQA: n_kv_heads < n_heads).
    pub n_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Attention type.
    pub attention_type: AttentionType,
    /// Sliding window size (0 = full attention).
    pub sliding_window: usize,
    /// Reference token count for R-SWA (always visible, never evicted).
    pub reference_tokens: usize,
    /// RoPE theta (base frequency for positional encoding).
    pub rope_theta: f32,
    /// Whether to apply QK normalization (per-head RMS norm).
    pub qk_norm: bool,
    /// Softcap for logits (0.0 = no capping, e.g. Gemma uses 50.0).
    pub softcap: f32,
    /// Partial rotary factor (fraction of head_dim to rotate, e.g. 0.25 for Ornith).
    pub partial_rotary_factor: f32,
}

impl AttentionConfig {
    /// GQA ratio: how many query heads per KV head group.
    pub fn gqa_ratio(&self) -> usize {
        if self.n_kv_heads == 0 {
            return 1;
        }
        self.n_heads / self.n_kv_heads
    }

    /// Attention scale factor (1/sqrt(head_dim)).
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Whether this is full attention (no sliding window).
    pub fn is_full_attention(&self) -> bool {
        self.sliding_window == 0
    }

    /// Total Q projection size.
    pub fn q_size(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Total K/V projection size.
    pub fn kv_size(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            attention_type: AttentionType::GQA,
            sliding_window: 0,
            reference_tokens: 0,
            rope_theta: 500000.0,
            qk_norm: false,
            softcap: 0.0,
            partial_rotary_factor: 1.0,
        }
    }
}
