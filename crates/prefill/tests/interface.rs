//! Unit tests for `engine_mlx_prefill::interface` module.

use engine_mlx_prefill::interface::Interface;

#[test]
fn interface_new_bails_with_mlx_marker() {
    let err = Interface::new().err().expect("expected bail");
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
