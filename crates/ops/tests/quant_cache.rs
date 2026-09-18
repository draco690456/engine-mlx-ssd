//! Unit tests for `engine_mlx_ops::quant_cache` module.
//!
//! `QuantCache` construction is stubbed and panics without mlx-c.

use engine_mlx_ops::ffi::mlx_array;

fn dummy_array() -> mlx_array {
    mlx_array(std::ptr::null_mut())
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn quant_cache_new_panics_without_mlx() {
    let _ = engine_mlx_ops::quant_cache::QuantCache::new(1024, 128, 0);
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn quant_cache_append_panics_without_mlx() {
    let mut cache = engine_mlx_ops::quant_cache::QuantCache::new(1024, 128, 0);
    cache.append(dummy_array(), dummy_array());
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn quant_cache_get_panics_without_mlx() {
    let cache = engine_mlx_ops::quant_cache::QuantCache::new(1024, 128, 0);
    let _ = cache.get(0, 10);
}
