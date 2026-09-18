//! Unit tests for `engine_mlx_attention::gated` module (gated attention).

use engine_mlx_attention::gated;
use engine_mlx_ops::MlxCtx;
use engine_mlx_ops::ffi::mlx_array;

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}

fn dummy_weights() -> engine_mlx_attention::QuantWeights {
    engine_mlx_attention::QuantWeights {
        weight: mlx_array(std::ptr::null_mut()),
        scales: mlx_array(std::ptr::null_mut()),
        biases: mlx_array(std::ptr::null_mut()),
        group_size: 64,
        bits: 4,
        mode: "linear".to_string(),
    }
}

#[test]
fn project_q_gated_bails_with_mlx_marker() {
    let err = gated::project_q_gated(
        &null_ctx(),
        mlx_array(std::ptr::null_mut()),
        &dummy_weights(),
        &dummy_weights(),
        32,
        128,
        1024,
    )
    .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

#[test]
fn apply_output_gate_bails_with_mlx_marker() {
    let err = gated::apply_output_gate(
        &null_ctx(),
        mlx_array(std::ptr::null_mut()),
        mlx_array(std::ptr::null_mut()),
    )
    .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
