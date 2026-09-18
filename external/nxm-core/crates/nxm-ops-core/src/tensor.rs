//! Core tensor types: Tensor, DType, Shape.
//!
//! These are CPU-backed value types used for trait boundaries.
//! GPU tensors are engine-internal and never cross crate boundaries.

use std::fmt;

/// Data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    U32,
    U8,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::U8 => 1,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I32 => write!(f, "i32"),
            DType::U32 => write!(f, "u32"),
            DType::U8 => write!(f, "u8"),
        }
    }
}

/// Tensor shape — dynamic dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    /// Scalar (rank 0).
    pub fn scalar() -> Self {
        Self(vec![])
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.0.iter().product::<usize>().max(1)
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Get dimension at index.
    pub fn dim(&self, index: usize) -> usize {
        self.0[index]
    }

    /// Get dimensions as slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

/// CPU-backed tensor — used at trait boundaries.
///
/// GPU tensors are engine-internal. When data crosses crate boundaries
/// (e.g., from engine to server), it travels as this CPU Tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    bytes: Vec<u8>,
    shape: Shape,
    dtype: DType,
}

impl Tensor {
    /// Create tensor from raw bytes.
    pub fn from_bytes(bytes: Vec<u8>, shape: Shape, dtype: DType) -> Self {
        debug_assert_eq!(
            bytes.len(),
            shape.numel() * dtype.size_bytes(),
            "Tensor byte length mismatch: {} bytes for shape {} dtype {}",
            bytes.len(),
            shape,
            dtype
        );
        Self { bytes, shape, dtype }
    }

    /// Create tensor from f32 slice.
    pub fn from_f32(data: &[f32], shape: Shape) -> Self {
        let bytes = data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Self {
            bytes,
            shape,
            dtype: DType::F32,
        }
    }

    /// Create tensor from u32 slice.
    pub fn from_u32(data: &[u32], shape: Shape) -> Self {
        let bytes = data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        Self {
            bytes,
            shape,
            dtype: DType::U32,
        }
    }

    /// Create a zero-filled tensor.
    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        let bytes = vec![0u8; shape.numel() * dtype.size_bytes()];
        Self { bytes, shape, dtype }
    }

    /// View as f32 slice (panics if dtype != F32).
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32, "Tensor is not f32");
        unsafe {
            std::slice::from_raw_parts(
                self.bytes.as_ptr() as *const f32,
                self.bytes.len() / 4,
            )
        }
    }

    /// View as mutable f32 slice (panics if dtype != F32).
    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.dtype, DType::F32, "Tensor is not f32");
        unsafe {
            std::slice::from_raw_parts_mut(
                self.bytes.as_mut_ptr() as *mut f32,
                self.bytes.len() / 4,
            )
        }
    }

    /// View as u32 slice (panics if dtype != U32).
    pub fn as_u32_slice(&self) -> &[u32] {
        assert_eq!(self.dtype, DType::U32, "Tensor is not u32");
        unsafe {
            std::slice::from_raw_parts(
                self.bytes.as_ptr() as *const u32,
                self.bytes.len() / 4,
            )
        }
    }

    /// Raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Into raw bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Shape reference.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bytes.len()
    }
}
