//! Unit tests for `engine_mlx_kvcache::fp8` (FP8 E4M3FN cache, simulated).
use engine_mlx_kvcache::KvCache;
use engine_mlx_kvcache::Fp8Cache;

#[test]
fn new_is_empty() {
    let cache = Fp8Cache::new();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.memory_bytes(), 0);
}

#[test]
fn reset_keeps_invariant() {
    let mut cache = Fp8Cache::new();
    cache.reset();
    assert!(cache.is_empty());
}

#[test]
fn append_bails_with_mlx_marker() {
    let mut cache = Fp8Cache::new();
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
    let cache = Fp8Cache::new();
    let err = cache.get(&null_ctx()).unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}
