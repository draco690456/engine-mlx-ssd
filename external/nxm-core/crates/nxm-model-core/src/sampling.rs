//! Sampling trait and configuration.

/// Token sampler — selects next token from logits.
///
/// Implementations: GreedySampler, TopKSampler, Mirostat2Sampler, etc.
pub trait Sampler: Send {
    /// Sample a token index from logits.
    ///
    /// `logits`: [vocab_size] raw unnormalized scores
    /// Returns: selected token index (0..vocab_size)
    fn sample(&mut self, logits: &[f32]) -> usize;

    /// Reset sampler state (for stateful samplers like Mirostat).
    fn reset(&mut self) {}
}

/// Sampling configuration — parameters for the sampler.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature scaling (0.0 = greedy, 1.0 = neutral).
    pub temperature: f32,
    /// Top-k: keep only top-k logits (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus): keep logits summing to p probability mass.
    pub top_p: f32,
    /// Min-p: discard tokens with prob < min_p * max_prob.
    pub min_p: f32,
    /// Repetition penalty factor.
    pub repetition_penalty: f32,
    /// Frequency penalty (penalize based on occurrence count).
    pub frequency_penalty: f32,
    /// Presence penalty (penalize if token appeared at all).
    pub presence_penalty: f32,
    /// Mirostat target entropy (0.0 = disabled).
    pub mirostat_tau: f32,
    /// Mirostat learning rate.
    pub mirostat_eta: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 0,
            top_p: 0.9,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat_tau: 0.0,
            mirostat_eta: 0.1,
        }
    }
}
