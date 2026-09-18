//! Unit tests for `engine_mlx_serve::server` module.

use engine_mlx_serve::server::Server;

#[test]
fn server_new_bails_with_mlx_marker() {
    let err = Server::new().err().expect("expected bail");
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
