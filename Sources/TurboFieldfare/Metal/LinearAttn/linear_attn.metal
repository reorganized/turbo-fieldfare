#include <metal_stdlib>
using namespace metal;

// ============================================================================
// linear_attn.metal — Qwen3.6 hybrid kernels: gated-delta-rule linear
// attention (Qwen3-Next GDN) plus the small layer epilogues its decoder
// layers need. Reference math: mlx-lm mlx_lm/models/qwen3_5.py +
// mlx_lm/models/gated_delta.py (fp32 state, scalar per-v-head gating).
//
// All activations are FP16 in/out with FP32 arithmetic inside kernels, and
// T carries the token count so decode (T=1) and prefill share one kernel.
// ============================================================================

static inline float qwen_silu(float x) {
    return x / (1.0f + fast::exp(-x));
}

static inline float qwen_softplus(float x) {
    // log1p(exp(x)), overflow-safe above x ≈ 20 (matches nn.softplus).
    return x > 20.0f ? x : log(1.0f + fast::exp(x));
}

// Two-stage block sum reduce: per-SIMD simd_sum into threadgroup partials,
// then SIMD group 0 merges and broadcasts through partial[0].
static inline float la_block_reduce_sum(
    float acc,
    uint  lid,
    uint  simd_lane_id,
    uint  simd_group_id,
    uint  simdgroups,
    threadgroup float* partial
) {
    acc = simd_sum(acc);
    if (simdgroups > 1) {
        if (simd_lane_id == 0) partial[simd_group_id] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
            float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
            v = simd_sum(v);
            if (simd_lane_id == 0) partial[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partial[0];
}

// ----------------------------------------------------------------------------
// Depthwise causal conv1d + SiLU over the fused qkv channels.
//
//   convInput rows: [window (kernel-1 rows), qkv (T rows)]
//   convOut[t, c]   = silu(Σ_j w[c, j] * convInput[t + j, c])
//
// One thread per (token, channel). Weight is BF16 [C, kernel, 1] contiguous.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_conv1d_silu(
    device const half*  qkv        [[buffer(0)]],   // [T, C]
    device const half*  convWindow [[buffer(1)]],   // [kernel - 1, C]
    device const bfloat* weight     [[buffer(2)]],   // [C, kernel, 1]
    device       half*  convOut     [[buffer(3)]],   // [T, C]
    constant     uint&  T           [[buffer(4)]],
    constant     uint&  C           [[buffer(5)]],
    constant     uint&  kernelSize  [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]            // (token, channel)
) {
    const uint t = gid.x;
    const uint c = gid.y;
    if (t >= T || c >= C) return;

    float acc = 0.0f;
    for (uint j = 0; j < kernelSize; ++j) {
        const uint inputRow = t + j;               // over (kernel-1) + T rows
        float v;
        if (inputRow < kernelSize - 1) {
            v = float(convWindow[inputRow * C + c]);
        } else {
            v = float(qkv[(inputRow - (kernelSize - 1)) * C + c]);
        }
        acc = fma(float(weight[c * kernelSize + j]), v, acc);
    }
    convOut[t * C + c] = half(qwen_silu(acc));
}

// ----------------------------------------------------------------------------
// Roll the conv window forward: newWindow[w] = convInput[T + w] over the
// concatenated (window, qkv) input, i.e. the last (kernel - 1) rows.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_conv_window_update(
    device const half*  qkv        [[buffer(0)]],   // [T, C]
    device const half*  oldWindow  [[buffer(1)]],   // [kernel - 1, C]
    device       half*  newWindow  [[buffer(2)]],   // [kernel - 1, C]
    constant     uint&  T           [[buffer(3)]],
    constant     uint&  C           [[buffer(4)]],
    constant     uint&  kernelSize  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]            // (window row, channel)
) {
    const uint w = gid.x;
    const uint c = gid.y;
    if (w >= kernelSize - 1 || c >= C) return;
    const uint inputRow = T + w;                    // over (kernel-1) + T rows
    float v;
    if (inputRow < kernelSize - 1) {
        v = float(oldWindow[inputRow * C + c]);
    } else {
        v = float(qkv[(inputRow - (kernelSize - 1)) * C + c]);
    }
    newWindow[w * C + c] = half(v);
}

// ----------------------------------------------------------------------------
// normalize_qk: l2-normalize q and k per head (eps added to the SUM of
// squares) and fold the Dk^-0.5 readout scale into q — the reference
// normalize_qk ends with |k| = 1 and |q| = Dk^-0.5. q occupies columns
// [0, Hk*Dk) of the conv output, k occupies [Hk*Dk, 2*Hk*Dk); in place.
// Dispatch: one threadgroup per (head, token) covering headDim lanes.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_l2norm_qk(
    device       half* convOut [[buffer(0)]],       // [T, rowStride] (q|k in [0, 2*Hk*Dk))
    constant     uint& T        [[buffer(1)]],
    constant     uint& numHeads [[buffer(2)]],      // Hk
    constant     uint& headDim  [[buffer(3)]],      // Dk
    constant     uint& rowStride [[buffer(5)]],     // elements per token row
    constant     float& eps     [[buffer(4)]],
    uint  tg  [[threadgroup_position_in_grid]],      // flattened (head, token)
    uint  lid [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups [[simdgroups_per_threadgroup]]
) {
    const uint heads = 2 * numHeads;
    const uint h = tg % heads;
    const uint t = tg / heads;
    if (h >= heads || t >= T) return;
    threadgroup float partial[8];

    device half* head = convOut + t * rowStride + h * headDim;

    float acc = 0.0f;
    for (uint i = lid; i < headDim; i += lsize) {
        float v = float(head[i]);
        acc = fma(v, v, acc);
    }
    acc = la_block_reduce_sum(acc, lid, simd_lane_id, simd_group_id,
                              simdgroups, partial);
    const float inv = rsqrt(acc + eps);
    // The reference folds the readout scale into q only: |k| = 1, and
    // |q| = Dk^-0.5 so the recurrence's readout y = S·q carries the
    // attention scale.
    const float readout = (h < numHeads)
        ? rsqrt(float(headDim))
        : 1.0f;
    const float scale = inv * readout;
    for (uint i = lid; i < headDim; i += lsize) {
        head[i] = half(float(head[i]) * scale);
    }
}

// ----------------------------------------------------------------------------
// Gated delta rule recurrence over T tokens, fp32 state.
//
// Per v-head hv (key head hk = hv / (Hv/Hk)), state row [Dv, Dk] fp32:
//   g    = exp(-exp(A_log[hv]) * softplus(a[t, hv] + dt_bias[hv]))
//   beta = sigmoid(b[t, hv])
//   S    *= g
//   kv   = Σ_dk S[dv, dk] * k[t, hk, dk]
//   delta = (v[t, hv, dv] - kv) * beta
//   S[dv] += k[t, hk] * delta
//   y[t, hv, dv] = Σ_dk S[dv, dk] * q[t, hk, dk]
//
// Dispatch: one threadgroup per (v-head, dv row); 32 lanes each own
// Dk/32 state elements held in registers for the whole T loop — no barriers
// (every state row is owned by exactly one threadgroup; reductions are
// SIMD-only). State is read once and written once.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(32)]]
void qwen_gdn_recurse(
    device const half*  q        [[buffer(0)]],     // [T, Hk*Dk]
    device const half*  k        [[buffer(1)]],     // [T, Hk*Dk]
    device const half*  v        [[buffer(2)]],     // [T, Hv*Dv]
    device const half*  a        [[buffer(3)]],     // [T, Hv]
    device const half*  b        [[buffer(4)]],     // [T, Hv]
    device const bfloat* a_log   [[buffer(5)]],     // [Hv]
    device const bfloat* dt_bias [[buffer(6)]],     // [Hv]
    device       float* state    [[buffer(7)]],     // [Hv, Dv, Dk] fp32 (in/out)
    device       half*  y        [[buffer(8)]],     // [T, Hv*Dv]
    constant     uint&  T        [[buffer(9)]],
    constant     uint&  Hk       [[buffer(10)]],
    constant     uint&  Hv       [[buffer(11)]],
    constant     uint&  Dk       [[buffer(12)]],
    constant     uint&  Dv       [[buffer(13)]],
    constant     uint&  aStride  [[buffer(14)]],     // elements between tokens in a/b
    constant     uint&  qkvStride [[buffer(15)]],    // elements between tokens in q/k/v
    constant     uint&  yStride  [[buffer(16)]],     // elements between tokens in y
    uint  tg  [[threadgroup_position_in_grid]],      // flattened (hv, dv)
    uint  lane [[thread_index_in_simdgroup]]
) {
    if (Dk != 128) return;                          // register layout: 4/lane
    const uint hv = tg / Dv;
    const uint dv = tg % Dv;
    if (hv >= Hv || dv >= Dv) return;
    const uint hk = hv / (Hv / Hk);
    constexpr uint elemsPerLane = 4;

    float s[elemsPerLane];
    const uint stateBase = (hv * Dv + dv) * Dk;
    for (uint i = 0; i < elemsPerLane; ++i) {
        s[i] = state[stateBase + elemsPerLane * lane + i];
    }

    const float A = exp(float(a_log[hv]));
    const float dtb = float(dt_bias[hv]);

    for (uint t = 0; t < T; ++t) {
        const float g = exp(-A * qwen_softplus(
            float(a[t * aStride + hv]) + dtb));
        const float beta = 1.0f / (1.0f + fast::exp(-float(b[t * aStride + hv])));

        device const half* kt = k + t * qkvStride + hk * Dk;
        device const half* qt = q + t * qkvStride + hk * Dk;
        device const half* vt = v + t * qkvStride + hv * Dv;

        float kvMem = 0.0f;
        for (uint i = 0; i < elemsPerLane; ++i) {
            const uint idx = elemsPerLane * lane + i;
            s[i] *= g;
            kvMem = fma(s[i], float(kt[idx]), kvMem);
        }
        kvMem = simd_sum(kvMem);
        const float delta = (float(vt[dv]) - kvMem) * beta;

        float out = 0.0f;
        for (uint i = 0; i < elemsPerLane; ++i) {
            const uint idx = elemsPerLane * lane + i;
            s[i] = fma(float(kt[idx]), delta, s[i]);
            out = fma(s[i], float(qt[idx]), out);
        }
        out = simd_sum(out);
        if (lane == 0) {
            y[t * yStride + hv * Dv + dv] = half(out);
        }
    }

    for (uint i = 0; i < elemsPerLane; ++i) {
        state[stateBase + elemsPerLane * lane + i] = s[i];
    }
}

// ----------------------------------------------------------------------------
// Gated output norm: out = silu(z) * rms_norm(y, w) per (token, v-head).
// Reference Qwen3NextRMSNormGated: rms over the value dims with eps in the
// mean, then silu(z) in fp32 times the normed value, rounded once to fp16.
// Dispatch: (Hv, T), threadgroup covers Dv lanes.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_gdn_gated_norm(
    device const half*  y        [[buffer(0)]],     // [T, Hv*Dv]
    device const half*  z        [[buffer(1)]],     // [T, Hv*Dv]
    device const bfloat* weight  [[buffer(2)]],     // [Dv]
    device       half*  out      [[buffer(3)]],     // [T, Hv*Dv]
    constant     uint&  T        [[buffer(4)]],
    constant     uint&  Hv       [[buffer(5)]],
    constant     uint&  Dv       [[buffer(6)]],
    constant     float& eps     [[buffer(7)]],
    uint  tg  [[threadgroup_position_in_grid]],      // flattened (head, token)
    uint  lid [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups [[simdgroups_per_threadgroup]]
) {
    const uint h = tg % Hv;
    const uint t = tg / Hv;
    if (h >= Hv || t >= T) return;
    threadgroup float partial[8];

    device const half* yh = y + t * Hv * Dv + h * Dv;
    device const half* zh = z + t * Hv * Dv + h * Dv;
    device half* oh = out + t * Hv * Dv + h * Dv;

    float acc = 0.0f;
    for (uint i = lid; i < Dv; i += lsize) {
        float v = float(yh[i]);
        acc = fma(v, v, acc);
    }
    acc = la_block_reduce_sum(acc, lid, simd_lane_id, simd_group_id,
                              simdgroups, partial);
    const float inv = rsqrt(acc / float(Dv) + eps);
    for (uint i = lid; i < Dv; i += lsize) {
        const float gate = qwen_silu(float(zh[i]));
        oh[i] = half(gate * (float(yh[i]) * inv * float(weight[i])));
    }
}

// ----------------------------------------------------------------------------
// Qwen3.6 full-attention Q/K post-processing, fused:
//   - per-head RMSNorm on q and k (BF16 weights, eps in mean),
//   - NeoX RoPE over rotatedPairs pairs with frequency divisor =
//     rotatedPairs*2 (the partial-rotary convention: exponent -2i/rotaryDim),
//   - sigmoid output gate fused into q: q[i] *= sigmoid(gate[i]).
//
// q rows are [numQHeads * headDim * 2] laid out per head as
// [q headDim | gate headDim] (the reference chunk(2, dim=-1) on the per-head
// view), with `qInStride` elements between tokens. The gated per-head q
// values are written to `qOut` at `qOutStride` elements per token (decode
// aliases qOut = q). K rows are [numKVHeads * headDim], normalized and
// rotated in place. V is untouched (Qwen3.6 has no v norm).
// Dispatch: (numQHeads + numKVHeads, T), 256 threads per threadgroup.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_attn_epilogue(
    device const half*  q            [[buffer(0)]],
    device       half*  qOut         [[buffer(12)]],
    constant     uint&  qInStride    [[buffer(13)]],
    constant     uint&  qOutStride   [[buffer(14)]],
    device       half*  k            [[buffer(1)]],
    device const bfloat* q_weight     [[buffer(2)]],  // [headDim]
    device const bfloat* k_weight     [[buffer(3)]],  // [headDim]
    constant     uint&  startPosition [[buffer(4)]],
    constant     uint&  T            [[buffer(5)]],
    constant     uint&  headDim      [[buffer(6)]],
    constant     uint&  numQHeads    [[buffer(7)]],
    constant     uint&  numKVHeads   [[buffer(8)]],
    constant     uint&  rotatedPairs [[buffer(9)]],
    constant     float& thetaBase    [[buffer(10)]],
    constant     float& rmsEps       [[buffer(11)]],
    uint  tg  [[threadgroup_position_in_grid]],      // flattened (head, token)
    uint  lid [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups [[simdgroups_per_threadgroup]]
) {
    const uint heads = numQHeads + numKVHeads;
    const uint h = tg % heads;
    const uint t = tg / heads;
    if (t >= T) return;
    threadgroup float partial[8];
    threadgroup half tgQ[256];
    threadgroup half tgGate[256];

    const uint pairCount = rotatedPairs;
    const uint freqDivisor = rotatedPairs * 2;
    const float position = float(startPosition + t);

    if (h < numQHeads) {
        device const half* inRow = q + t * qInStride + h * headDim * 2;
        device half* outRow = qOut + t * qOutStride + h * headDim;
        float acc = 0.0f;
        for (uint i = lid; i < headDim; i += lsize) {
            tgQ[i] = inRow[i];
            tgGate[i] = inRow[headDim + i];
            float v = float(tgQ[i]);
            acc = fma(v, v, acc);
        }
        acc = la_block_reduce_sum(acc, lid, simd_lane_id, simd_group_id,
                                  simdgroups, partial);
        const float inv = rsqrt(acc / float(headDim) + rmsEps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < headDim; i += lsize) {
            tgQ[i] = half(float(tgQ[i]) * inv * float(q_weight[i]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < pairCount) {
            // Partial NeoX rotary: rotaryDim = 2*pairCount dims, pairs
            // (i, i + pairCount) for i < pairCount; dims beyond the rotary
            // section are left untouched.
            const uint halfRot = pairCount;
            const float x0 = float(tgQ[lid]);
            const float x1 = float(tgQ[halfRot + lid]);
            const float angle = position
                * pow(thetaBase, -float(2u * lid) / float(freqDivisor));
            const float c = cos(angle);
            const float s = sin(angle);
            tgQ[lid] = half(x0 * c - x1 * s);
            tgQ[halfRot + lid] = half(x0 * s + x1 * c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < headDim; i += lsize) {
            outRow[i] = tgQ[i];
        }
    } else {
        const uint kh = h - numQHeads;
        if (kh >= numKVHeads) return;
        device half* row = k + t * numKVHeads * headDim + kh * headDim;
        float acc = 0.0f;
        for (uint i = lid; i < headDim; i += lsize) {
            tgQ[i] = row[i];
            float v = float(tgQ[i]);
            acc = fma(v, v, acc);
        }
        acc = la_block_reduce_sum(acc, lid, simd_lane_id, simd_group_id,
                                  simdgroups, partial);
        const float inv = rsqrt(acc / float(headDim) + rmsEps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < headDim; i += lsize) {
            tgQ[i] = half(float(tgQ[i]) * inv * float(k_weight[i]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < pairCount) {
            const uint halfRot = pairCount;
            const float x0 = float(tgQ[lid]);
            const float x1 = float(tgQ[halfRot + lid]);
            const float angle = position
                * pow(thetaBase, -float(2u * lid) / float(freqDivisor));
            const float c = cos(angle);
            const float s = sin(angle);
            tgQ[lid] = half(x0 * c - x1 * s);
            tgQ[halfRot + lid] = half(x0 * s + x1 * c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < headDim; i += lsize) {
            row[i] = tgQ[i];
        }
    }
}

// ----------------------------------------------------------------------------
// Post-attention setup: h += r in place, then normed2 = rms_norm(h, w).
// Qwen3.6 has a single FFN input norm (no sandwich, no router norm quirk).
// Dispatch: one threadgroup per token row.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_post_attn_setup(
    device       half*  h        [[buffer(0)]],      // [T, D] (in/out)
    device const half*  r        [[buffer(1)]],      // [T, D]
    device       half*  normed2  [[buffer(2)]],      // [T, D]
    device const bfloat* weight   [[buffer(3)]],      // [D]
    constant     uint&  T        [[buffer(4)]],
    constant     uint&  D        [[buffer(5)]],
    constant     float& eps      [[buffer(6)]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  simd_lane_id [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]],
    uint  simdgroups [[simdgroups_per_threadgroup]],
    uint  t [[threadgroup_position_in_grid]]
) {
    if (t >= T) return;
    device half* ht = h + t * D;
    device const half* rt = r + t * D;
    threadgroup float partial[8];

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = float(ht[i]) + float(rt[i]);
        ht[i] = half(v);
        acc = fma(v, v, acc);
    }
    acc = la_block_reduce_sum(acc, lid, simd_lane_id, simd_group_id,
                              simdgroups, partial);
    const float inv = rsqrt(acc / float(D) + eps);
    device half* o = normed2 + t * D;
    for (uint i = lid; i < D; i += lsize) {
        o[i] = half(float(ht[i]) * inv * float(weight[i]));
    }
}

// ----------------------------------------------------------------------------
// Layer tail: hidden = h + sharedGated + routed. Qwen3.6 has no layer scalar
// and no output norms on the branch outputs.
// Dispatch: one threadgroup per token row.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_layer_tail(
    device       half*  hidden [[buffer(0)]],           // [T, D] (in/out)
    device const half*  h1     [[buffer(1)]],           // [T, D] shared, already gated
    device const half*  h2     [[buffer(2)]],           // [T, D] routed
    constant     uint&  T      [[buffer(3)]],
    constant     uint&  D      [[buffer(4)]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  t [[threadgroup_position_in_grid]]
) {
    if (t >= T) return;
    device half* ht = hidden + t * D;
    device const half* a = h1 + t * D;
    device const half* b = h2 + t * D;
    for (uint i = lid; i < D; i += lsize) {
        ht[i] = half(float(ht[i]) + float(a[i]) + float(b[i]));
    }
}

// ----------------------------------------------------------------------------
// Shared-expert scalar gate: sharedOut row t *= sigmoid(gateLogit[t]).
// Dispatch: one threadgroup per token row.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_shared_gate_mul(
    device       half*  sharedOut [[buffer(0)]],      // [T, D] (in/out)
    device const half*  gateLogit [[buffer(1)]],      // [T]
    constant     uint&  T         [[buffer(2)]],
    constant     uint&  D         [[buffer(3)]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  t [[threadgroup_position_in_grid]]
) {
    if (t >= T) return;
    const float g = 1.0f / (1.0f + fast::exp(-float(gateLogit[t])));
    device half* o = sharedOut + t * D;
    for (uint i = lid; i < D; i += lsize) {
        o[i] = half(float(o[i]) * g);
    }
}
// ----------------------------------------------------------------------------
// Sigmoid output gate for full-attention layers: attn[h*HD+i] *=
// sigmoid(gate[h*(2*HD) + HD + i]). The Qwen3-Next reference applies the
// gate to the attention output before o_proj, and the checkpoint stores
// the gate interleaved per head as [q HD | gate HD] blocks of the q_proj
// output. Dispatch: one threadgroup per token row.
// ----------------------------------------------------------------------------
[[kernel, max_total_threads_per_threadgroup(256)]]
void qwen_attn_gate(
    device       half*  attn      [[buffer(0)]],     // [T, NQ*HD] (in/out)
    device const half*  gate      [[buffer(1)]],     // [T, 2*NQ*HD] interleaved
    constant     uint&  T         [[buffer(2)]],
    constant     uint&  numHeads  [[buffer(3)]],
    constant     uint&  headDim   [[buffer(4)]],
    constant     uint&  gateStride [[buffer(5)]],
    uint  lid [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  t [[threadgroup_position_in_grid]]
) {
    if (t >= T) return;
    device half* o = attn + t * numHeads * headDim;
    device const half* g = gate + t * gateStride;
    for (uint i = lid; i < numHeads * headDim; i += lsize) {
        const uint h = i / headDim;
        const uint d = i % headDim;
        const float gv = float(g[h * 2 * headDim + headDim + d]);
        o[i] = half(float(o[i]) * (1.0f / (1.0f + fast::exp(-gv))));
    }
}
