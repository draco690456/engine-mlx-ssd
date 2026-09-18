//! KV Cache traits.
//!
//! **NOTE**: Stub for workspace structure.

use crate::ffi::mlx_array;

pub trait KvCacheBackend: Send + Sync {
    fn append(&mut self, k: mlx_array, v: mlx_array) -> anyhow::Result<()>;
    fn get(&self, start: usize, len: usize) -> anyhow::Result<(mlx_array, mlx_array)>;
    fn capacity(&self) -> usize;
    fn len(&self) -> usize;
}

pub trait TensorRef: Send + Sync {
    fn as_ptr(&self) -> *const std::ffi::c_void;
    fn shape(&self) -> &[usize];
    fn dtype(&self) -> u32;
}
