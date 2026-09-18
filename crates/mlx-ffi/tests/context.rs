//! Unit tests for `engine_mlx_ffi::context` (safe stream/array wrappers).

use engine_mlx_ffi::{MlxContext, mlx_stream};

fn null_stream() -> mlx_stream {
    mlx_stream(std::ptr::null_mut())
}

#[test]
fn new_wraps_stream_without_calling_mlx() {
    let ctx = MlxContext::new(null_stream());
    // check() is pure logic, no MLX call.
    MlxContext::check(0, "noop").unwrap();
}

#[test]
fn check_propagates_nonzero_codes() {
    let _ctx = MlxContext::new(null_stream());
    let err = MlxContext::check(7, "matmul").unwrap_err();
    assert!(err.to_string().contains("matmul"));
    assert!(err.to_string().contains("7"));
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn cpu_panics_without_mlx() {
    let _ = MlxContext::cpu();
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn gpu_panics_without_mlx() {
    let _ = MlxContext::gpu();
}
