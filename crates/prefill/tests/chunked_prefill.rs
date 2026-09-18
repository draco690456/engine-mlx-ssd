//! Unit tests for `engine_mlx_prefill::chunked_prefill` module.

use engine_mlx_ops::MlxCtx;
use engine_mlx_prefill::chunked_prefill::chunked_prefill;

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}

#[test]
fn chunked_prefill_bails_with_mlx_marker() {
    let err = chunked_prefill(&null_ctx()).unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
