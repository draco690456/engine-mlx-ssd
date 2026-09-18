//! MLX-C FFI re-exports.
//!
//! **NOTE**: Stub for workspace structure.

pub use engine_mlx_ffi::*;

// Re-export common types
pub type mlx_array = engine_mlx_ffi::mlx_array;
pub type mlx_stream = engine_mlx_ffi::mlx_stream;
pub type mlx_dtype = engine_mlx_ffi::mlx_dtype;
pub type mlx_optional_int_ = engine_mlx_ffi::mlx_optional_int_;
pub type mlx_optional_float_ = engine_mlx_ffi::mlx_optional_float_;
pub type mlx_optional_dtype_ = engine_mlx_ffi::mlx_optional_dtype_;

// Re-export common functions
pub use engine_mlx_ffi::{
    mlx_default_cpu_stream_new, mlx_default_gpu_stream_new,
    mlx_synchronize, mlx_synchronize_stream,
    mlx_array_new_data, mlx_array_free, mlx_array_eval, mlx_eval,
    mlx_quantized_matmul, mlx_fast_rope, mlx_fast_rope_dynamic,
    mlx_rms_norm, mlx_layer_norm, mlx_silu, mlx_softmax,
    mlx_multiply, mlx_add, mlx_matmul, mlx_take, mlx_astype,
    mlx_zeros, mlx_full, mlx_arange,
};
