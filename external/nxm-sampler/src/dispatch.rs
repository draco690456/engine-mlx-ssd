//! GPU sampler dispatch — keeps everything on Metal, reads back only 4 bytes.

use anyhow::{anyhow, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ptr::NonNull;

use crate::config::{SamplerConfig, SamplerStrategy};

/// GPU-resident sampler. Pre-compiles pipelines at creation.
pub struct GpuSampler {
    config: SamplerConfig,
    vocab_size: u32,
    // Pre-compiled pipelines
    pipeline_softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_topk: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_argmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_rep_penalty: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    // Pre-allocated buffers (zero alloc in hot path)
    probs_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    result_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    threshold_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl GpuSampler {
    /// Create sampler with pre-compiled pipelines and pre-allocated buffers.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        vocab_size: u32,
        config: SamplerConfig,
    ) -> Result<Self> {
        let source = include_str!("../shaders/sampler.metal");
        let ns_source = NSString::from_str(source);
        let library = device.newLibraryWithSource_options_error(&ns_source, None)
            .map_err(|e| anyhow!("Compile sampler shaders: {e}"))?;

        let make_pipeline = |name: &str| -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
            let ns_name = NSString::from_str(name);
            let func = library.newFunctionWithName(&ns_name)
                .ok_or_else(|| anyhow!("Function '{}' not found", name))?;
            device.newComputePipelineStateWithFunction_error(&func)
                .map_err(|e| anyhow!("Pipeline '{}': {e}", name))
        };

        let pipeline_softmax = make_pipeline("softmax_temperature")?;
        let pipeline_topk = make_pipeline("top_k_filter")?;
        let pipeline_argmax = make_pipeline("argmax_sample")?;
        let pipeline_rep_penalty = make_pipeline("repetition_penalty")?;

        // Pre-allocate output buffers (never reallocated)
        let probs_buffer = device.newBufferWithLength_options(
            (vocab_size as usize) * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("Alloc probs buffer"))?;
        let result_buffer = device.newBufferWithLength_options(
            4, MTLResourceOptions::StorageModeShared)  // single u32
            .ok_or_else(|| anyhow!("Alloc result buffer"))?;
        let threshold_buffer = device.newBufferWithLength_options(
            4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| anyhow!("Alloc threshold buffer"))?;

        Ok(Self {
            config,
            vocab_size,
            pipeline_softmax,
            pipeline_topk,
            pipeline_argmax,
            pipeline_rep_penalty,
            probs_buffer,
            result_buffer,
            threshold_buffer,
        })
    }

    /// Sample a token from logits — entirely on GPU.
    ///
    /// `logits_buffer` must be a [vocab_size] f32 MTLBuffer.
    /// Returns the sampled token ID (single u32 readback = 4 bytes).
    ///
    /// Call this INSIDE the same command encoder as the forward pass
    /// for zero-overhead integration.
    pub fn sample_in_encoder(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        logits_buffer: &ProtocolObject<dyn MTLBuffer>,
        context_buffer: Option<&ProtocolObject<dyn MTLBuffer>>,
        context_len: u32,
    ) {
        let tg_size = MTLSize { width: 256, height: 1, depth: 1 };
        let grid = MTLSize { width: 1, height: 1, depth: 1 };

        // Step 1: Repetition penalty (if context provided)
        if let Some(ctx) = context_buffer {
            if self.config.repetition_penalty != 1.0 && context_len > 0 {
                unsafe {
                    encoder.setComputePipelineState(&self.pipeline_rep_penalty);
                    encoder.setBuffer_offset_atIndex(Some(logits_buffer), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(ctx), 0, 1);
                    let ctx_len_bytes = context_len.to_le_bytes();
                    encoder.setBytes_length_atIndex(NonNull::new(ctx_len_bytes.as_ptr() as *mut _).unwrap(), 4, 2);
                    let rep_pen_bytes = self.config.repetition_penalty.to_le_bytes();
                    encoder.setBytes_length_atIndex(NonNull::new(rep_pen_bytes.as_ptr() as *mut _).unwrap(), 4, 3);
                    let rep_grid = MTLSize { width: ((context_len as usize + 255) / 256), height: 1, depth: 1 };
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(rep_grid, tg_size);
                }
                encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            }
        }

        // Step 2: Top-K filter (if not greedy)
        if self.config.strategy == SamplerStrategy::TopK || self.config.strategy == SamplerStrategy::TopKP {
            unsafe {
                encoder.setComputePipelineState(&self.pipeline_topk);
                encoder.setBuffer_offset_atIndex(Some(logits_buffer), 0, 0);
                let vocab_bytes = self.vocab_size.to_le_bytes();
                encoder.setBytes_length_atIndex(NonNull::new(vocab_bytes.as_ptr() as *mut _).unwrap(), 4, 1);
                let topk_bytes = self.config.top_k.to_le_bytes();
                encoder.setBytes_length_atIndex(NonNull::new(topk_bytes.as_ptr() as *mut _).unwrap(), 4, 2);
                encoder.setBuffer_offset_atIndex(Some(&self.threshold_buffer), 0, 3);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg_size);
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }

        // Step 3: Greedy (argmax) or temperature+softmax
        match self.config.strategy {
            SamplerStrategy::Greedy => {
                unsafe {
                    encoder.setComputePipelineState(&self.pipeline_argmax);
                    encoder.setBuffer_offset_atIndex(Some(logits_buffer), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&self.result_buffer), 0, 1);
                    let vocab_bytes = self.vocab_size.to_le_bytes();
                    encoder.setBytes_length_atIndex(NonNull::new(vocab_bytes.as_ptr() as *mut _).unwrap(), 4, 2);
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg_size);
            }
            _ => {
                // Temperature + softmax
                unsafe {
                    encoder.setComputePipelineState(&self.pipeline_softmax);
                    encoder.setBuffer_offset_atIndex(Some(logits_buffer), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&self.probs_buffer), 0, 1);
                    let vocab_bytes = self.vocab_size.to_le_bytes();
                    encoder.setBytes_length_atIndex(NonNull::new(vocab_bytes.as_ptr() as *mut _).unwrap(), 4, 2);
                    let temp_bytes = self.config.temperature.to_le_bytes();
                    encoder.setBytes_length_atIndex(NonNull::new(temp_bytes.as_ptr() as *mut _).unwrap(), 4, 3);
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg_size);
                encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);

                // Then argmax on probs (TODO: categorical sampling with random)
                unsafe {
                    encoder.setComputePipelineState(&self.pipeline_argmax);
                    encoder.setBuffer_offset_atIndex(Some(&self.probs_buffer), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&self.result_buffer), 0, 1);
                    let vocab_bytes = self.vocab_size.to_le_bytes();
                    encoder.setBytes_length_atIndex(NonNull::new(vocab_bytes.as_ptr() as *mut _).unwrap(), 4, 2);
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg_size);
            }
        }
    }

    /// Read the sampled token back from GPU (4 bytes only!).
    pub fn read_result(&self) -> u32 {
        let ptr = self.result_buffer.contents().as_ptr() as *const u32;
        unsafe { *ptr }
    }

    /// Update config (e.g., change temperature mid-conversation).
    pub fn set_config(&mut self, config: SamplerConfig) {
        self.config = config;
    }
}