//! Unit tests for `engine_mlx_serve::config` module.

use engine_mlx_serve::config::Config;

#[test]
fn config_new_bails_with_mlx_marker() {
    let err = Config::new().err().expect("expected bail");
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
