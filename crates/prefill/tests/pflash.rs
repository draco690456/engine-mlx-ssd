//! Unit tests for `engine_mlx_prefill::pflash` module.

use engine_mlx_ops::MlxCtx;
use engine_mlx_prefill::pflash::PFlash;

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}

#[test]
fn pflash_new_bails_with_mlx_marker() {
    let err = PFlash::new(&null_ctx()).err().expect("expected bail");
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
