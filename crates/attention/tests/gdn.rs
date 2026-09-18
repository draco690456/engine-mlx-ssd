//! Unit tests for `engine_mlx_attention::gdn` module (Gated Delta Network).

use engine_mlx_attention::{gdn, GdnConfig, GdnState};
use engine_mlx_ops::MlxCtx;
use engine_mlx_ops::ffi::mlx_array;

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}

fn config() -> GdnConfig {
    GdnConfig::new(1024, 16, 64)
}

#[test]
fn gdn_config_new_sets_fields() {
    let cfg = config();
    assert_eq!(cfg.hidden_dim, 1024);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.head_dim, 64);
}

#[test]
fn gdn_compute_gate_bails_with_mlx_marker() {
    let err = gdn::compute_gate(&null_ctx(), mlx_array(std::ptr::null_mut()), &config())
        .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

#[test]
fn gdn_step_bails_with_mlx_marker() {
    let mut state = GdnState { h: None };
    let err = gdn::gdn_step(&null_ctx(), &mut state, mlx_array(std::ptr::null_mut()), &config())
        .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}

#[test]
fn gdn_forward_bails_with_mlx_marker() {
    let err = gdn::gdn_forward(&null_ctx(), mlx_array(std::ptr::null_mut()), &config())
        .unwrap_err();
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
