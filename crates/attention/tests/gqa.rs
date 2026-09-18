//! Unit tests for `engine_mlx_attention::gqa` module.
//!
//! All GQA helpers are stubbed and bail with the mlx-c marker. The `MlxCtx`
//! cannot be constructed without MLX, so a zeroed (Copy) handle is used; the
//! stub functions never dereference it before bailing.

use engine_mlx_attention::gqa;
use engine_mlx_ops::MlxCtx;
use engine_mlx_ops::ffi::mlx_array;

fn null_ctx() -> MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}

fn dummy_array() -> mlx_array {
    mlx_array(std::ptr::null_mut())
}

fn dummy_weights() -> engine_mlx_attention::QuantWeights {
    engine_mlx_attention::QuantWeights {
        weight: dummy_array(),
        scales: dummy_array(),
        biases: dummy_array(),
        group_size: 64,
        bits: 4,
        mode: "linear".to_string(),
    }
}

#[test]
fn project_qkv_bails_with_mlx_marker() {
    let err = gqa::project_qkv(
        &null_ctx(),
        dummy_array(),
        &dummy_weights(),
        &dummy_weights(),
        &dummy_weights(),
        32,
        8,
        128,
        1024,
    )
    .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

#[test]
fn optional_qk_norm_bails_with_mlx_marker() {
    let err = gqa::optional_qk_norm(&null_ctx(), dummy_array(), dummy_array(), None, None)
        .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

#[test]
fn transpose_for_attn_bails_with_mlx_marker() {
    let err =
        gqa::transpose_for_attn(&null_ctx(), dummy_array(), dummy_array(), dummy_array())
            .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

#[test]
fn kv_cache_concat_bails_with_mlx_marker() {
    let err = gqa::kv_cache_concat(
        &null_ctx(),
        None,
        dummy_array(),
        None,
        dummy_array(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

#[test]
fn attend_and_project_bails_with_mlx_marker() {
    let err = gqa::attend_and_project(
        &null_ctx(),
        dummy_array(),
        dummy_array(),
        dummy_array(),
        &dummy_weights(),
        1.0,
        true,
    )
    .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
