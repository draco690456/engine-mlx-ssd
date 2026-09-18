//! Quantized weight types and traits.
//!
//! Defines the interface for quantized matrix multiplication
//! used across all backends (CPU dequant, MLX qmatmul, Metal Q4 kernel).

use crate::tensor::DType;

/// Quantization mode for weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantMode {
    /// No quantization (full precision).
    None,
    /// 4-bit quantization (group-wise).
    Q4,
    /// 8-bit quantization (group-wise).
    Q8,
    /// 2-bit quantization (OSCAR-style).
    Q2,
    /// 5-bit quantization.
    Q5,
    /// 6-bit quantization.
    Q6,
    /// FP8 E4M3FN (for KV cache).
    FP8,
    /// MXFP4 (microscaling).
    MXFP4,
    /// MXFP8 (microscaling).
    MXFP8,
    /// NVFP4 (NVIDIA format).
    NVFP4,
}

impl QuantMode {
    /// Bits per element.
    pub fn bits(&self) -> usize {
        match self {
            QuantMode::None => 32,
            QuantMode::Q2 => 2,
            QuantMode::Q4 | QuantMode::MXFP4 | QuantMode::NVFP4 => 4,
            QuantMode::Q5 => 5,
            QuantMode::Q6 => 6,
            QuantMode::Q8 | QuantMode::FP8 | QuantMode::MXFP8 => 8,
        }
    }

    /// Compression ratio vs FP32.
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bits() as f32
    }
}

/// Quantized weights bundle — everything needed for qmatmul.
///
/// Layout depends on backend:
/// - CPU: packed bytes + scales + biases, dequant on-the-fly
/// - MLX: mlx_array handle (opaque)
/// - Metal: MTLBuffer with u4-packed data + scale/bias buffers
#[derive(Debug, Clone)]
pub struct QuantWeights {
    /// Packed quantized weight data.
    pub data: Vec<u8>,
    /// Per-group scale factors (f16 or f32 depending on backend).
    pub scales: Vec<u8>,
    /// Per-group bias/zero-point (f16 or f32).
    pub biases: Vec<u8>,
    /// Number of bits per element.
    pub bits: u32,
    /// Group size for quantization.
    pub group_size: u32,
    /// Quantization mode.
    pub mode: QuantMode,
    /// Output dimension (N in [N, K] weight matrix).
    pub out_features: usize,
    /// Input dimension (K in [N, K] weight matrix).
    pub in_features: usize,
    /// Data type of scales/biases.
    pub scale_dtype: DType,
}

impl QuantWeights {
    /// Estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + self.scales.len() + self.biases.len()
    }

    /// Whether this weight is actually quantized (vs full precision).
    pub fn is_quantized(&self) -> bool {
        self.mode != QuantMode::None
    }
}
