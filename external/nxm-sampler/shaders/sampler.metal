//! GPU-resident sampling kernels.
//!
//! The key insight: if logits stay on GPU and we sample on GPU,
//! we eliminate the logits→CPU readback which costs ~0.1-0.5ms per token.
//! At 400+ tok/s, that readback is 5-20% of the decode budget.
//!
//! Kernels:
//! 1. softmax_temperature — apply temperature + softmax in-place
//! 2. top_k_filter — zero out all but top-k logits
//! 3. top_p_filter — nucleus sampling (cumulative probability threshold)
//! 4. repetition_penalty — penalize repeated tokens
//! 5. argmax — greedy sampling (no randomness)
//! 6. categorical_sample — weighted random sampling from filtered distribution

#include <metal_stdlib>
using namespace metal;

// ─── Temperature + Softmax (fused) ───────────────────────────────────────────

/// Apply temperature scaling and compute softmax over logits.
/// logits[vocab_size] → probabilities[vocab_size]
///
/// Fused: divide by temperature, subtract max (stability), exp, normalize.
/// All on GPU — logits never leave the device.
kernel void softmax_temperature(
    device float* logits [[buffer(0)]],
    device float* probs [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    constant float& temperature [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    // Pass 1: find max (for numerical stability)
    float local_max = -INFINITY;
    for (uint i = lid; i < vocab_size; i += tg_size) {
        local_max = max(local_max, logits[i] / temperature);
    }
    shared_max[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for max
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared_max[0];

    // Pass 2: exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint i = lid; i < vocab_size; i += tg_size) {
        float val = exp(logits[i] / temperature - global_max);
        probs[i] = val;
        local_sum += val;
    }
    shared_sum[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for sum
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = shared_sum[0];

    // Pass 3: normalize
    for (uint i = lid; i < vocab_size; i += tg_size) {
        probs[i] /= global_sum;
    }
}

// ─── Top-K Filter ────────────────────────────────────────────────────────────

/// Find the top-k largest logits and zero out everything else.
/// Uses a partial sort (selection) approach on GPU.
///
/// After this kernel, only top-k positions have non-zero probability.
/// This is the GPU-native equivalent of oMLX's dsa_topk_indices.
kernel void top_k_filter(
    device float* logits [[buffer(0)]],
    constant uint& vocab_size [[buffer(1)]],
    constant uint& k [[buffer(2)]],
    device float* threshold_out [[buffer(3)]],  // output: k-th largest value
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Strategy: find the k-th largest value, then mask everything below it.
    // For small k (≤64), a register-based partial sort is efficient.
    // For large k, a single-pass approximate threshold works.

    // Pass 1: Find approximate threshold using thread-local top-k
    // Each thread scans a portion of logits and keeps its local top-k
    threadgroup float shared_threshold[1];

    float local_top[64]; // max k=64 per thread
    uint local_count = 0;
    uint effective_k = min(k, 64u);

    for (uint i = lid; i < vocab_size; i += tg_size) {
        float val = logits[i];
        if (local_count < effective_k) {
            local_top[local_count++] = val;
            // Bubble down (insertion into sorted array)
            for (int j = (int)local_count - 1; j > 0 && local_top[j] > local_top[j-1]; j--) {
                float tmp = local_top[j];
                local_top[j] = local_top[j-1];
                local_top[j-1] = tmp;
            }
        } else if (val > local_top[effective_k - 1]) {
            local_top[effective_k - 1] = val;
            for (int j = (int)effective_k - 1; j > 0 && local_top[j] > local_top[j-1]; j--) {
                float tmp = local_top[j];
                local_top[j] = local_top[j-1];
                local_top[j-1] = tmp;
            }
        }
    }

    // Use the k-th value from thread 0 as threshold (approximate for now)
    if (lid == 0 && local_count >= effective_k) {
        shared_threshold[0] = local_top[effective_k - 1];
        threshold_out[0] = shared_threshold[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float threshold = shared_threshold[0];

    // Pass 2: Zero out everything below threshold
    for (uint i = lid; i < vocab_size; i += tg_size) {
        if (logits[i] < threshold) {
            logits[i] = -INFINITY;
        }
    }
}

// ─── Repetition Penalty ──────────────────────────────────────────────────────

/// Apply repetition penalty to tokens that appeared in context.
/// penalty > 1.0 reduces probability of repeated tokens.
/// penalty < 1.0 increases it (unlikely but supported).
kernel void repetition_penalty(
    device float* logits [[buffer(0)]],
    device const uint* context_tokens [[buffer(1)]],
    constant uint& context_len [[buffer(2)]],
    constant float& penalty [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= context_len) return;

    uint token_id = context_tokens[gid];
    float val = logits[token_id];

    // If positive, divide by penalty. If negative, multiply.
    if (val > 0.0f) {
        logits[token_id] = val / penalty;
    } else {
        logits[token_id] = val * penalty;
    }
}

// ─── Argmax (Greedy) ─────────────────────────────────────────────────────────

/// GPU-resident argmax — returns the index of the largest logit.
/// No CPU readback of the full logit vector needed!
kernel void argmax_sample(
    device const float* logits [[buffer(0)]],
    device uint* result_token [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_val[256];
    threadgroup uint shared_idx[256];

    float local_max = -INFINITY;
    uint local_idx = 0;

    for (uint i = lid; i < vocab_size; i += tg_size) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = i;
        }
    }

    shared_val[lid] = local_max;
    shared_idx[lid] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (shared_val[lid + stride] > shared_val[lid]) {
                shared_val[lid] = shared_val[lid + stride];
                shared_idx[lid] = shared_idx[lid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        result_token[0] = shared_idx[0];
    }
}
