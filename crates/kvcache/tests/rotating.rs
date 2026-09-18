//! Unit tests for `engine_mlx_kvcache::rotating` (sliding-window cache).
use engine_mlx_kvcache::KvCache;
use engine_mlx_kvcache::RotatingCache;

#[test]
fn new_is_empty() {
    let cache = RotatingCache::new();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.memory_bytes(), 0);
}

#[test]
fn reset_keeps_invariant() {
    let mut cache = RotatingCache::new();
    cache.reset();
    assert!(cache.is_empty());
}

#[test]
fn append_bails_with_mlx_marker() {
    let mut cache = RotatingCache::new();
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
    let cache = RotatingCache::new();
    let err = cache.get(&null_ctx()).unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}
