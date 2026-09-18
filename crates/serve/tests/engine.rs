//! Unit tests for `engine_mlx_serve::engine` module.

use engine_mlx_serve::engine::Engine;

#[test]
fn engine_new_bails_with_mlx_marker() {
    let err = Engine::new().err().expect("expected bail");
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
