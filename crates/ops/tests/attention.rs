//! Unit tests for `engine_mlx_ops::attention` module.
//!
//! The ops crate is a stub: MLX-C FFI is not bound, so atomic attention
//! operations panic with a clear "MLX feature not enabled" message.
//! These tests lock that contract so the real binding cannot silently break it.

use engine_mlx_ops::MlxCtx;
use engine_mlx_ops::ffi::mlx_array;

fn dummy_array() -> mlx_array {
    mlx_array(std::ptr::null_mut())
}

fn null_ctx() -> MlxCtx {
    MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn gqa_attention_panics_without_mlx() {
    let ctx = null_ctx();
    let q = dummy_array();
    let k = dummy_array();
    let v = dummy_array();
    let _ = engine_mlx_ops::attention::gqa_attention(&ctx, q, k, v, 1.0, None);
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn sliding_window_attention_panics_without_mlx() {
    let ctx = null_ctx();
    let q = dummy_array();
    let k = dummy_array();
    let v = dummy_array();
    let _ = engine_mlx_ops::attention::sliding_window_attention(&ctx, q, k, v, 512, 1.0, None);
}
