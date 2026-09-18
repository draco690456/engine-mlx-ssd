//! Attention mask types.

/// Mask for attention computation.
///
/// Controls which tokens can attend to which.
#[derive(Debug, Clone)]
pub enum Mask {
    /// Standard causal mask: token i can attend to tokens [0..=i].
    Causal,

    /// Sliding window causal: token i attends to [max(0, i-window)..=i].
    SlidingCausal {
        /// Window size in tokens.
        window: usize,
    },

    /// R-SWA: Reference tokens (always visible) + sliding decode window.
    ///
    /// Token i attends to:
    /// - All reference tokens [0..ref_len] (always)
    /// - Sliding window [max(ref_len, i-window)..=i]
    ReferenceSlidingWindow {
        /// Number of reference (prompt) tokens — always visible.
        reference_len: usize,
        /// Sliding window size for decode tokens.
        window: usize,
    },

    /// No mask (full bidirectional attention — for encoding/embedding).
    None,

    /// Custom binary mask (true = attend, false = block).
    ///
    /// Shape: [query_len, key_len]. Row i, col j = whether query i
    /// can attend to key j.
    Custom(Vec<bool>),
}

impl Mask {
    /// Whether query position `q` can attend to key position `k`.
    pub fn can_attend(&self, q: usize, k: usize, _total_keys: usize) -> bool {
        match self {
            Mask::Causal => k <= q,
            Mask::SlidingCausal { window } => {
                k <= q && (q - k) < *window
            }
            Mask::ReferenceSlidingWindow {
                reference_len,
                window,
            } => {
                if k < *reference_len {
                    // Reference tokens always visible
                    true
                } else {
                    // Decode tokens: sliding window
                    k <= q && (q - k) < *window
                }
            }
            Mask::None => true,
            Mask::Custom(data) => {
                // Assume data is [query_len * key_len] row-major
                // This is a simplified check
                data.get(q * _total_keys + k).copied().unwrap_or(false)
            }
        }
    }

    /// Maximum number of keys a single query can attend to.
    ///
    /// Used for memory allocation and performance estimation.
    pub fn max_attend_count(&self, total_keys: usize) -> usize {
        match self {
            Mask::Causal | Mask::None => total_keys,
            Mask::SlidingCausal { window } => (*window).min(total_keys),
            Mask::ReferenceSlidingWindow {
                reference_len,
                window,
            } => (*reference_len + *window).min(total_keys),
            Mask::Custom(_) => total_keys,
        }
    }
}
