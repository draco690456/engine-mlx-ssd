//! Sampler configuration.

/// Sampling strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplerStrategy {
    /// Always pick the highest probability token.
    Greedy,
    /// Keep only top-K tokens, sample from them.
    TopK,
    /// Keep tokens until cumulative probability ≥ P.
    TopP,
    /// Top-K then Top-P (combined).
    TopKP,
}

/// Sampler configuration.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Temperature for logit scaling (0.0 = greedy, 1.0 = neutral).
    pub temperature: f32,
    /// Top-K: keep only K highest logits.
    pub top_k: u32,
    /// Top-P (nucleus): keep until cumulative prob ≥ p.
    pub top_p: f32,
    /// Repetition penalty (>1.0 = penalize repeats).
    pub repetition_penalty: f32,
    /// Sampling strategy.
    pub strategy: SamplerStrategy,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            strategy: SamplerStrategy::TopK,
        }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, strategy: SamplerStrategy::Greedy, ..Default::default() }
    }
}
