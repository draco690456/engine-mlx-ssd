//! Unit tests for `engine_mlx_kvcache::turboquant` (Lloyd-Max codebook cache).
use engine_mlx_kvcache::KvCache;
use engine_mlx_kvcache::{TurboQuantCache, TurboQuantConfig};

#[test]
fn config_default_is_4bit_group64() {
    let cfg = TurboQuantConfig::default();
    assert_eq!(cfg.bits, 4);
    assert_eq!(cfg.group_size, 64);
}

#[test]
fn cache_new_is_empty() {
    let cache = TurboQuantCache::new();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.memory_bytes(), 0);
}

#[test]
fn reset_keeps_invariant() {
    let mut cache = TurboQuantCache::new();
    cache.reset();
    assert!(cache.is_empty());
}

#[test]
fn append_bails_with_mlx_marker() {
    let mut cache = TurboQuantCache::new();
    let err = cache
        .append(
            &null_ctx(),
            engine_mlx_ops::ffi::mlx_array(std::ptr::null_mut()),
            engine_mlx_ops::ffi::mlx_array(std::ptr::null_mut()),
        )
        .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

#[test]
fn get_bails_with_mlx_marker() {
    let cache = TurboQuantCache::new();
    let err = cache.get(&null_ctx()).unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}
