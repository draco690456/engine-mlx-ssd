//! Cache-specific types: TensorRef, CompressedTensor, CacheConfig, BlockAddr.

/// Reference to tensor data in various formats (compressed or raw).
///
/// Used at the cache trait boundary. Backends store in their preferred
/// format internally; this enum is for inter-crate communication.
#[derive(Debug, Clone)]
pub enum TensorRef {
    /// Full precision f32 data.
    F32(Vec<f32>),
    /// Half precision f16 data (stored as raw bytes).
    F16(Vec<u8>),
    /// BFloat16 data (stored as raw bytes).
    BF16(Vec<u8>),
    /// Compressed (quantized) tensor with metadata.
    Compressed(CompressedTensor),
}

impl TensorRef {
    /// Number of logical elements.
    pub fn len(&self) -> usize {
        match self {
            TensorRef::F32(d) => d.len(),
            TensorRef::F16(d) => d.len() / 2,
            TensorRef::BF16(d) => d.len() / 2,
            TensorRef::Compressed(c) => c.numel,
        }
    }

    /// Whether this reference contains no data.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        match self {
            TensorRef::F32(d) => d.len() * 4,
            TensorRef::F16(d) | TensorRef::BF16(d) => d.len(),
            TensorRef::Compressed(c) => c.data.len(),
        }
    }
}

/// Compressed (quantized) tensor representation.
#[derive(Debug, Clone)]
pub struct CompressedTensor {
    /// Compressed data bytes.
    pub data: Vec<u8>,
    /// Number of logical elements (before compression).
    pub numel: usize,
    /// Quantization type used.
    pub quant_type: QuantType,
    /// Per-group codebooks or scale factors (if needed).
    pub codebook: Option<Vec<f32>>,
}

/// Quantization type for KV cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantType {
    /// No quantization — pass-through.
    None,
    /// INT8 symmetric (scale per group).
    Int8,
    /// TurboQuant: FWHT rotation + Lloyd-Max codebook (6.4× compression).
    TurboQuant,
    /// Lloyd-Max scalar quantization (4×).
    LloydMax,
    /// IsoQuant: quaternion rotation + quantization (4×).
    IsoQuant,
    /// RotorQuant: Clifford rotor rotation + quantization (4×).
    RotorQuant,
    /// PlanarQuant: Givens rotation + 4-bit Lloyd-Max (4×).
    PlanarQuant,
    /// ParoQuant: FWHT + Givens rotation + quantization (4×).
    ParoQuant,
    /// FP8 E4M3FN (for inline KV cache compression).
    FP8,
    /// 2-bit OSCAR quantization (8× compression).
    Oscar2Bit,
}

impl QuantType {
    /// Approximate compression ratio vs FP32.
    pub fn compression_ratio(&self) -> f32 {
        match self {
            QuantType::None => 1.0,
            QuantType::Int8 | QuantType::FP8 => 4.0,
            QuantType::LloydMax | QuantType::IsoQuant
            | QuantType::RotorQuant | QuantType::PlanarQuant
            | QuantType::ParoQuant => 4.0,
            QuantType::TurboQuant => 6.4,
            QuantType::Oscar2Bit => 8.0,
        }
    }
}

/// Configuration for a cache instance.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of tokens to cache.
    pub max_tokens: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// Number of KV heads per layer.
    pub n_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Quantization type for this cache.
    pub quant_type: QuantType,
    /// Block size for paged allocation (0 = no paging).
    pub block_size: usize,
}

/// Block address for paged KV cache (à la oMLX).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockAddr {
    /// Physical block index.
    pub block_idx: u32,
    /// Number of valid tokens in this block.
    pub n_valid: u16,
    /// Layer index.
    pub layer: u16,
}

/// Configuration for attend_compressed operation.
#[derive(Debug, Clone)]
pub struct AttendConfig {
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of KV heads.
    pub n_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Attention scale (usually 1/sqrt(head_dim)).
    pub scale: f32,
}
