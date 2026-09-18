//! Unit tests for `engine_mlx_ops::compile` module.
//!
//! Graph capture is stubbed: `Closure::execute` panics, while `compile`
//! and `compile_shapeless` panic at construction.

use engine_mlx_ops::compile::{compile, compile_shapeless};
use engine_mlx_ops::MlxCtx;

fn null_ctx() -> MlxCtx {
    MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn compile_panics_without_mlx() {
    let _ = compile(&null_ctx(), || {});
}

#[test]
#[should_panic(expected = "MLX feature not enabled")]
fn compile_shapeless_panics_without_mlx() {
    let _ = compile_shapeless(&null_ctx(), || {});
}
