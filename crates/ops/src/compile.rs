//! MLX-C compilation utilities.
//!
//! **NOTE**: Stub for workspace structure.

use crate::ffi::mlx_array;
use crate::MlxCtx;

pub struct Closure {
    _private: (),
}

impl Closure {
    pub fn execute(&self, _inputs: &[mlx_array]) -> anyhow::Result<Vec<mlx_array>> {
        panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c.")
    }
}

pub fn compile(_ctx: &MlxCtx, _f: impl FnOnce() + 'static) -> anyhow::Result<Closure> {
    panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c.")
}

pub fn compile_shapeless(_ctx: &MlxCtx, _f: impl FnOnce() + 'static) -> anyhow::Result<Closure> {
    panic!("MLX feature not enabled. Compile with --features mlx or install mlx-c.")
}
