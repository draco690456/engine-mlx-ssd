//! Unit tests for `engine_mlx_serve::tokenizer` module.

use engine_mlx_serve::tokenizer::Tokenizer;

#[test]
fn tokenizer_new_bails_with_mlx_marker() {
    let err = Tokenizer::new().err().expect("expected bail");
    assert!(err.to_string().contains("needs mlx-c FFI"));
}
