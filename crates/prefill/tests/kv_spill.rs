//! Unit tests for `engine_mlx_prefill::kv_spill` module.

use engine_mlx_ops::MlxCtx;
use engine_mlx_prefill::kv_spill::KvSpill;

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}

#[test]
fn kv_spill_new_bails_with_mlx_marker() {
    let err = KvSpill::new(&null_ctx()).err().expect("expected bail");
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
