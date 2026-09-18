//! Model and engine configuration types.

/// Model hyperparameters — loaded from config.json.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub max_position_embeddings: usize,
    /// Number of experts (0 = dense model).
    pub num_experts: usize,
    /// Number of active experts per token (top-k).
    pub num_experts_per_token: usize,
    /// Whether the model has shared/always-active expert(s).
    pub has_shared_expert: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_layers: 28,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            vocab_size: 151936,
            intermediate_size: 8960,
            rms_norm_eps: 1e-6,
            rope_theta: 500000.0,
            sliding_window: 0,
            max_position_embeddings: 131072,
            num_experts: 0,
            num_experts_per_token: 0,
            has_shared_expert: false,
        }
    }
}

/// Engine type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EngineKind {
    /// Apple MLX (compiled graph via mlx-c).
    Mlx,
    /// C++ compiled bridge (single forward function).
    BridgeC,
    /// Pure Metal compute shaders (no MLX).
    MetalNative,
    /// CPU (candle/BLAS).
    Cpu,
    /// SSD streaming (MoE expert swap).
    Ssd,
    /// Auto-detect based on platform + model format.
    Auto,
}

/// Inference configuration (sampling + generation params).
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: usize,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Repetition penalty (1.0 = no penalty).
    pub repetition_penalty: f32,
    /// Min-p sampling threshold (0.0 = disabled).
    pub min_p: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            max_tokens: 1024,
            repetition_penalty: 1.0,
            min_p: 0.0,
        }
    }
}
