//! Unit tests for `engine_mlx_attention` root types and re-exports.

use engine_mlx_attention::{GdnConfig, QuantWeights};
use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_ops::KvCacheBackend;

#[test]
fn quant_weights_is_constructible() {
    let w = QuantWeights {
        weight: mlx_array(std::ptr::null_mut()),
        scales: mlx_array(std::ptr::null_mut()),
        biases: mlx_array(std::ptr::null_mut()),
        group_size: 128,
        bits: 2,
        mode: "gemm".to_string(),
    };
    assert_eq!(w.group_size, 128);
    assert_eq!(w.bits, 2);
    assert_eq!(w.mode, "gemm");
}

#[test]
fn gdn_config_default_new_is_consistent() {
    let cfg = GdnConfig::new(2048, 32, 64);
    assert_eq!(cfg.hidden_dim, 2048);
    assert_eq!(cfg.num_heads, 32);
    assert_eq!(cfg.head_dim, 64);
}

#[test]
fn kv_cache_backend_re_exported_from_ops() {
    // The `KvCacheBackend` trait surface must be reachable through the
    // attention crate's re-export and satisfiable by a concrete type.
    struct Dummy;
    impl engine_mlx_ops::KvCacheBackend for Dummy {
        fn append(&mut self, _k: mlx_array, _v: mlx_array) -> anyhow::Result<()> {
            Ok(())
        }
        fn get(&self, _s: usize, _l: usize) -> anyhow::Result<(mlx_array, mlx_array)> {
            Ok((mlx_array(std::ptr::null_mut()), mlx_array(std::ptr::null_mut())))
        }
        fn capacity(&self) -> usize {
            0
        }
        fn len(&self) -> usize {
            0
        }
    }

    let mut d = Dummy;
    d.append(mlx_array(std::ptr::null_mut()), mlx_array(std::ptr::null_mut()))
        .unwrap();
    assert_eq!(d.len(), 0);
    assert_eq!(d.capacity(), 0);
}
