//! Cross-crate integration tests for the MLX engine pipeline.
//!
//! These tests wire together `engine_mlx_ops`, `engine_mlx_attention`,
//! `engine_mlx_kvcache` and `engine_mlx_prefill` to verify the shared
//! `mlx_array` type flows across crate boundaries and that the stubbed
//! pipeline fails uniformly (same "needs mlx-c FFI" marker) before any real
//! MLX-C binding exists.
//!
//! The only hardware-dependent check (a real forward pass) is `#[ignore]`d
//! per RULES.md: it requires mlx-c headers and an Apple GPU.

use engine_mlx_attention::{gqa, GdnConfig, QuantWeights};
use engine_mlx_kvcache::{ConcatCache, KvCache};
use engine_mlx_ops::MlxCtx;
use engine_mlx_ops::ffi::mlx_array;
use engine_mlx_prefill::chunked_prefill::chunked_prefill;

fn null_ctx() -> engine_mlx_ops::MlxCtx {
    engine_mlx_ops::MlxCtx::new(engine_mlx_ops::ffi::mlx_stream(std::ptr::null_mut()))
}

fn dummy_array() -> mlx_array {
    mlx_array(std::ptr::null_mut())
}

fn dummy_weights() -> QuantWeights {
    QuantWeights {
        weight: dummy_array(),
        scales: dummy_array(),
        biases: dummy_array(),
        group_size: 64,
        bits: 4,
        mode: "linear".to_string(),
    }
}

/// `mlx_array` is the single FFI token shared by every engine crate.
/// This test fails to compile if the type diverges across crates.
#[test]
fn mlx_array_is_shared_across_crates() {
    // A tensor produced via ops is accepted wherever mlx_array is expected.
    let tensor: mlx_array = dummy_array();
    let weights = QuantWeights {
        weight: tensor,
        scales: dummy_array(),
        biases: dummy_array(),
        group_size: 64,
        bits: 4,
        mode: "linear".to_string(),
    };
    // kvcache accepts the same token shape via the KvCache trait surface.
    let mut cache = ConcatCache::new();
    cache
        .append(&null_ctx(), weights.weight, weights.scales)
        .ok(); // stubbed: errors, but the *type* is accepted by the compiler.
    let _cfg = GdnConfig::new(1024, 16, 64);
}

/// Every stage of the prefill->attention pipeline bails with the same marker,
/// so callers get a consistent "needs mlx-c FFI" contract across crates.
#[test]
fn pipeline_stages_fail_uniformly_without_mlx() {
    let ctx = null_ctx();

    let attn_err = gqa::project_qkv(
        &ctx,
        dummy_array(),
        &dummy_weights(),
        &dummy_weights(),
        &dummy_weights(),
        32,
        8,
        128,
        1024,
    )
    .unwrap_err();
    assert!(attn_err.to_string().contains("needs mlx-c FFI"));

    let mut cache = ConcatCache::new();
    let cache_err = cache
        .append(&ctx, dummy_array(), dummy_array())
        .unwrap_err();
    assert!(cache_err.to_string().contains("needs mlx-c FFI"));

    let prefill_err = chunked_prefill(&ctx).unwrap_err();
    assert!(prefill_err.to_string().contains("needs mlx-c FFI"));
}

/// Real end-to-end forward pass. Ignored: requires mlx-c headers and an Apple
/// GPU (#[ignore] per RULES.md — hardware-dependent tests must be opt-in).
#[test]
#[ignore = "requires mlx-c headers + Apple GPU (mlx feature enabled)"]
fn forward_pass_with_real_model() {
    let ctx = MlxCtx::gpu();
    let mut cache = ConcatCache::new();
    // Real flow would: load model -> prefill -> attention -> sample.
    let _ = (&ctx, &mut cache);
    unimplemented!("wire up a real Qwen3 forward pass once mlx-c FFI lands");
}
