//! Unit tests for `engine_mlx_serve::loader` module.

use engine_mlx_serve::loader::Loader;

#[test]
fn loader_new_bails_with_mlx_marker() {
    let err = Loader::new().err().expect("expected bail");
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
