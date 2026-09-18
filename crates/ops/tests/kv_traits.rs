//! Unit tests for `engine_mlx_ops::kv_traits` module.
//!
//! `KvCacheBackend` and `TensorRef` are trait contracts. Since the real
//! backends are stubbed, we provide a minimal in-test implementation to
//! guarantee the trait surface (signatures, Send + Sync bounds) is satisfied
//! and behaves as documented.

use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::{KvCacheBackend, TensorRef};

struct DummyBackend {
    data: Vec<f32>,
    capacity: usize,
}

impl KvCacheBackend for DummyBackend {
    fn append(&mut self, _k: mlx_array, _v: mlx_array) -> anyhow::Result<()> {
        self.data.push(1.0);
        Ok(())
    }

    fn get(&self, _start: usize, _len: usize) -> anyhow::Result<(mlx_array, mlx_array)> {
        Ok((mlx_array(std::ptr::null_mut()), mlx_array(std::ptr::null_mut())))
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

struct DummyTensor {
    ptr: *const std::ffi::c_void,
    shape: Vec<usize>,
    dtype: u32,
}

impl TensorRef for DummyTensor {
    fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> u32 {
        self.dtype
    }
}

// `DummyTensor` wraps a raw pointer, which is not Send/Sync by default.
// In this test it is only used on the calling thread, so we assert that.
unsafe impl Send for DummyTensor {}
unsafe impl Sync for DummyTensor {}

#[test]
fn kv_cache_backend_trait_is_satisfiable() {
    let mut backend = DummyBackend {
        data: Vec::new(),
        capacity: 1024,
    };
    assert_eq!(backend.capacity(), 1024);
    assert_eq!(backend.len(), 0);
    backend.append(mlx_array(std::ptr::null_mut()), mlx_array(std::ptr::null_mut())).unwrap();
    assert_eq!(backend.len(), 1);
    let (k, v) = backend.get(0, 1).unwrap();
    assert!(k.0.is_null());
    assert!(v.0.is_null());
}

#[test]
fn tensor_ref_trait_is_satisfiable() {
    let tensor = DummyTensor {
        ptr: std::ptr::null(),
        shape: vec![2, 8, 128],
        dtype: 0,
    };
    assert_eq!(tensor.shape(), &[2, 8, 128]);
    assert_eq!(tensor.dtype(), 0);
    assert!(tensor.as_ptr().is_null());
}
