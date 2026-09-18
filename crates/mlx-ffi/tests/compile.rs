//! Unit tests for `engine_mlx_ffi::compile` (graph capture).
//!
//! `compile`/`compile_shapeless` attempt to build an MLX context first, which
//! is stubbed, so they panic with the feature-gate message (the internal
//! "not bound yet" bail is currently unreachable under the stub).

use engine_mlx_ffi::compile::{compile, compile_shapeless};

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn compile_panics_without_mlx() {
    let _ = compile(|_ctx| Ok(()));
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn compile_shapeless_panics_without_mlx() {
    let _ = compile_shapeless(|_ctx| Ok(()));
}
