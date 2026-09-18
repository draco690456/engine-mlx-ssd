//! Low-level MLX-C FFI helpers for the forward pass.
//!
//! **NOTE**: Stub implementation. Full MLX-C FFI integration needed.

use anyhow::Result;
use crate::*;
use std::ffi::CString;

pub struct MlxCtx {
    pub stream: mlx_stream,
}

impl MlxCtx {
    pub fn new(stream: mlx_stream) -> Self {
        Self { stream }
    }

    pub fn cpu() -> Self {
        let stream = unsafe { mlx_default_cpu_stream_new() };
        Self { stream }
    }

    pub fn gpu() -> Self {
        let stream = unsafe { mlx_default_gpu_stream_new() };
        Self { stream }
    }

    pub fn check(val: i32, op: &str) -> Result<()> {
        if val != 0 {
            anyhow::bail!("MLX op '{}' failed with code {}", op, val);
        }
        Ok(())
    }

    // All ops stubbed - implement when mlx-c FFI is available
    pub fn embed(&self, _weights: mlx_array, _ids: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("embed not implemented - needs mlx-c FFI")
    }
    pub fn take(&self, _a: mlx_array, _indices: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("take not implemented - needs mlx-c FFI")
    }
    pub fn new_array_i32(&self, _data: &[i32]) -> Result<mlx_array> {
        anyhow::bail!("new_array_i32 not implemented - needs mlx-c FFI")
    }
    pub fn ndim(&self, _arr: mlx_array) -> Result<i32> {
        anyhow::bail!("ndim not implemented - needs mlx-c FFI")
    }
    pub fn expand_dims(&self, _x: mlx_array, _axis: i32) -> Result<mlx_array> {
        anyhow::bail!("expand_dims not implemented - needs mlx-c FFI")
    }
    pub fn copy_to_stream(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("copy_to_stream not implemented - needs mlx-c FFI")
    }
    pub fn arange(&self, _start: f32, _stop: f32, _step: f32) -> Result<mlx_array> {
        anyhow::bail!("arange not implemented - needs mlx-c FFI")
    }
    pub fn equal_scalar(&self, _a: mlx_array, _val: f32) -> Result<mlx_array> {
        anyhow::bail!("equal_scalar not implemented - needs mlx-c FFI")
    }
    pub fn rms_norm(&self, _x: mlx_array, _weight: mlx_array, _eps: f32) -> Result<mlx_array> {
        anyhow::bail!("rms_norm not implemented - needs mlx-c FFI")
    }
    pub fn matmul(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("matmul not implemented - needs mlx-c FFI")
    }
    pub fn add(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("add not implemented - needs mlx-c FFI")
    }
    pub fn multiply(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("multiply not implemented - needs mlx-c FFI")
    }
    pub fn silu(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("silu not implemented - needs mlx-c FFI")
    }
    pub fn conv1d(&self, _input: mlx_array, _weight: mlx_array, _stride: i32, _padding: i32, _dilation: i32, _groups: i32) -> Result<mlx_array> {
        anyhow::bail!("conv1d not implemented - needs mlx-c FFI")
    }
    pub fn reshape(&self, _x: mlx_array, _shape: &[i32]) -> Result<mlx_array> {
        anyhow::bail!("reshape not implemented - needs mlx-c FFI")
    }
    pub fn transpose(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("transpose not implemented - needs mlx-c FFI")
    }
    pub fn transpose_axes(&self, _x: mlx_array, _axes: &[i32]) -> Result<mlx_array> {
        anyhow::bail!("transpose_axes not implemented - needs mlx-c FFI")
    }
    pub fn rope(&self, _x: mlx_array, _dims: i32, _theta: f32, _offset: i32) -> Result<mlx_array> {
        anyhow::bail!("rope not implemented - needs mlx-c FFI")
    }
    pub fn rope_dynamic(&self, _x: mlx_array, _dims: i32, _theta: f32, _offset: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("rope_dynamic not implemented - needs mlx-c FFI")
    }
    pub fn sdpa(&self, _q: mlx_array, _k: mlx_array, _v: mlx_array, _scale: f32, _causal: bool) -> Result<mlx_array> {
        anyhow::bail!("sdpa not implemented - needs mlx-c FFI")
    }
    pub fn sdpa_masked(&self, _q: mlx_array, _k: mlx_array, _v: mlx_array, _scale: f32, _mask: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("sdpa_masked not implemented - needs mlx-c FFI")
    }
    pub fn tile(&self, _x: mlx_array, _reps: &[i32]) -> Result<mlx_array> {
        anyhow::bail!("tile not implemented - needs mlx-c FFI")
    }
    pub fn astype(&self, _x: mlx_array, _dtype: u32) -> Result<mlx_array> {
        anyhow::bail!("astype not implemented - needs mlx-c FFI")
    }
    pub fn dequantize_weight(&self, _w: mlx_array, _scales: mlx_array, _biases: mlx_array, _group_size: i32, _bits: i32) -> Result<mlx_array> {
        anyhow::bail!("dequantize_weight not implemented - needs mlx-c FFI")
    }
    pub fn dequantize_mxfp4(&self, _w: mlx_array, _scales: mlx_array, _group_size: i32) -> Result<mlx_array> {
        anyhow::bail!("dequantize_mxfp4 not implemented - needs mlx-c FFI")
    }
    pub fn quantized_matmul(&self, _x: mlx_array, _w: mlx_array, _scales: mlx_array, _biases: mlx_array, _transpose: bool, _group_size: i32, _bits: i32) -> Result<mlx_array> {
        anyhow::bail!("quantized_matmul not implemented - needs mlx-c FFI")
    }
    pub fn quantized_matmul_mode(&self, _x: mlx_array, _w: mlx_array, _scales: mlx_array, _biases: mlx_array, _transpose: bool, _group_size: i32, _bits: i32, _mode: &str) -> Result<mlx_array> {
        anyhow::bail!("quantized_matmul_mode not implemented - needs mlx-c FFI")
    }
    pub fn evaluate(&self, _arr: mlx_array) -> Result<()> {
        anyhow::bail!("evaluate not implemented - needs mlx-c FFI")
    }
    pub fn eval_and_sync(&self, _arr: mlx_array) -> Result<()> {
        anyhow::bail!("eval_and_sync not implemented - needs mlx-c FFI")
    }
    pub fn array_size(&self, _arr: mlx_array) -> usize {
        0
    }
    pub unsafe fn array_data_f32(&self, _arr: mlx_array) -> *const f32 {
        std::ptr::null()
    }
    pub fn eval_all(&self, _arrays: &[mlx_array]) -> Result<()> {
        anyhow::bail!("eval_all not implemented - needs mlx-c FFI")
    }
    pub fn eval_all_and_sync(&self, _arrays: &[mlx_array]) -> Result<()> {
        anyhow::bail!("eval_all_and_sync not implemented - needs mlx-c FFI")
    }
    pub fn materialize(&self, _arr: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("materialize not implemented - needs mlx-c FFI")
    }
    pub fn concatenate(&self, _a: mlx_array, _b: mlx_array, _axis: i32) -> Result<mlx_array> {
        anyhow::bail!("concatenate not implemented - needs mlx-c FFI")
    }
    pub fn split_sections(&self, _a: mlx_array, _sections: &[i32], _axis: i32) -> Result<Vec<mlx_array>> {
        anyhow::bail!("split_sections not implemented - needs mlx-c FFI")
    }
    pub fn stack(&self, _arrays: &[mlx_array], _axis: i32) -> Result<mlx_array> {
        anyhow::bail!("stack not implemented - needs mlx-c FFI")
    }
    pub fn slice(&self, _a: mlx_array, _start: &[i32], _stop: &[i32], _strides: &[i32]) -> Result<mlx_array> {
        anyhow::bail!("slice not implemented - needs mlx-c FFI")
    }
    pub fn slice_update(&self, _src: mlx_array, _update: mlx_array, _start: &[i32], _stop: &[i32], _strides: &[i32]) -> Result<mlx_array> {
        anyhow::bail!("slice_update not implemented - needs mlx-c FFI")
    }
    pub fn slice_update_dynamic(&self, _src: mlx_array, _update: mlx_array, _start: mlx_array, _axes: &[i32]) -> Result<mlx_array> {
        anyhow::bail!("slice_update_dynamic not implemented - needs mlx-c FFI")
    }
    pub fn shape(&self, _x: mlx_array) -> Result<Vec<i32>> {
        anyhow::bail!("shape not implemented - needs mlx-c FFI")
    }
    pub fn sigmoid(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("sigmoid not implemented - needs mlx-c FFI")
    }
    pub fn exp(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("exp not implemented - needs mlx-c FFI")
    }
    pub fn negative(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("negative not implemented - needs mlx-c FFI")
    }
    pub fn subtract(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("subtract not implemented - needs mlx-c FFI")
    }
    pub fn remainder(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("remainder not implemented - needs mlx-c FFI")
    }
    pub fn full_i32(&self, _shape: &[i32], _value: i32) -> Result<mlx_array> {
        anyhow::bail!("full_i32 not implemented - needs mlx-c FFI")
    }
    pub fn slice_dynamic(&self, _a: mlx_array, _start: mlx_array, _axes: &[i32], _size: &[i32]) -> Result<mlx_array> {
        anyhow::bail!("slice_dynamic not implemented - needs mlx-c FFI")
    }
    pub fn log1p(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("log1p not implemented - needs mlx-c FFI")
    }
    pub fn softplus(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("softplus not implemented - needs mlx-c FFI")
    }
    pub fn softmax(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("softmax not implemented - needs mlx-c FFI")
    }
    pub fn softmax_axis(&self, _x: mlx_array, _axis: i32) -> Result<mlx_array> {
        anyhow::bail!("softmax_axis not implemented - needs mlx-c FFI")
    }
    pub fn sum_axis(&self, _x: mlx_array, _axis: i32, _keep_dims: bool) -> Result<mlx_array> {
        anyhow::bail!("sum_axis not implemented - needs mlx-c FFI")
    }
    pub fn abs(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("abs not implemented - needs mlx-c FFI")
    }
    pub fn bridge_forward(&self, _token_id: i32, _offset: i32, _kv_keys: &mut [mlx_array], _kv_values: &mut [mlx_array], _rswa_window: i32) -> Result<mlx_array> {
        anyhow::bail!("bridge_forward not implemented - needs mlx-c FFI")
    }
    pub fn max(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("max not implemented - needs mlx-c FFI")
    }
    pub fn argmax(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("argmax not implemented - needs mlx-c FFI")
    }
    pub fn divide(&self, _a: mlx_array, _b: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("divide not implemented - needs mlx-c FFI")
    }
    pub fn round(&self, _x: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("round not implemented - needs mlx-c FFI")
    }
    pub fn new_array(&self, _data: &[f32]) -> Result<mlx_array> {
        anyhow::bail!("new_array not implemented - needs mlx-c FFI")
    }
    pub fn zeros(&self, _shape: &[i32]) -> Result<mlx_array> {
        anyhow::bail!("zeros not implemented - needs mlx-c FFI")
    }
    pub fn zeros_f32(&self, _shape: &[i32]) -> Result<mlx_array> {
        anyhow::bail!("zeros_f32 not implemented - needs mlx-c FFI")
    }
    pub fn zeros_typed(&self, _shape: &[i32], _like: mlx_array) -> Result<mlx_array> {
        anyhow::bail!("zeros_typed not implemented - needs mlx-c FFI")
    }
    pub fn full_f32(&self, _shape: &[i32], _value: f32) -> Result<mlx_array> {
        anyhow::bail!("full_f32 not implemented - needs mlx-c FFI")
    }
    pub fn to_vec_u32(&self, _arr: mlx_array) -> Result<Vec<u32>> {
        anyhow::bail!("to_vec_u32 not implemented - needs mlx-c FFI")
    }
    pub fn to_vec_f16_as_f32(&self, _arr: mlx_array) -> Result<Vec<f32>> {
        anyhow::bail!("to_vec_f16_as_f32 not implemented - needs mlx-c FFI")
    }
    pub fn to_vec_bf16_as_f32(&self, _arr: mlx_array) -> Result<Vec<f32>> {
        anyhow::bail!("to_vec_bf16_as_f32 not implemented - needs mlx-c FFI")
    }
    pub fn to_vec_f32(&self, _arr: mlx_array) -> Result<Vec<f32>> {
        anyhow::bail!("to_vec_f32 not implemented - needs mlx-c FFI")
    }
}