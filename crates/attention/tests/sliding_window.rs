//! Unit tests for `engine_mlx_attention::sliding_window` module.

use engine_mlx_attention::sliding_window;
use engine_mlx_ops::MlxCtx;
use engine_mlx_ops::ffi::mlx_array;

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}

#[test]
fn sliding_window_attention_bails_with_mlx_marker() {
    let err = sliding_window::sliding_window_attention(
        &null_ctx(),
        mlx_array(std::ptr::null_mut()),
        mlx_array(std::ptr::null_mut()),
        mlx_array(std::ptr::null_mut()),
        512,
        1.0,
    )
    .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
