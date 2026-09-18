//! MLX-C compile / graph capture utilities.

use anyhow::Result;
use crate::MlxCtx;

/// Compiled closure handle.
pub struct Closure {
    pub inner: mlx_closure,
}

impl Closure {
    pub fn new(inner: mlx_closure) -> Self {
        Self { inner }
    }
}

/// Compile a function into an MLX closure for faster execution.
///
/// The function `f` receives an `MlxCtx` and should perform MLX operations.
/// The returned `Closure` can be executed repeatedly with different inputs.
pub fn compile<F>(f: F) -> Result<Closure>
where
    F: FnOnce(&MlxCtx) -> Result<()>,
{
    let ctx = MlxCtx::gpu();
    f(&ctx)?;
    // TODO: Actual MLX compile API binding
    anyhow::bail!("mlx_compile not bound yet")
}

/// Compile a shapeless function (inputs can have dynamic shapes).
pub fn compile_shapeless<F>(f: F) -> Result<Closure>
where
    F: FnOnce(&MlxCtx) -> Result<()>,
{
    let ctx = MlxCtx::gpu();
    f(&ctx)?;
    // TODO: Actual MLX compile API binding
    anyhow::bail!("mlx_compile_shapeless not bound yet")
}

// FFI types for compile API (when available)
#[cfg(feature = "mlx")]
pub use mlx_sys::mlx_closure;

#[cfg(not(feature = "mlx"))]
mod stub_compile {
    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct mlx_closure(pub *mut std::ffi::c_void);
}

#[cfg(not(feature = "mlx"))]
pub use stub_compile::mlx_closure;