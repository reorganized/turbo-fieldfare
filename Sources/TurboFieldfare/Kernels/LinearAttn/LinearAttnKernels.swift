import Foundation
import Metal

/// Swift wrappers for the Qwen3.6 hybrid kernels (`linear_attn.metal`).
/// Decode passes `t: 1`; prefill passes the chunk token count.
final class LinearAttnKernels {

    private let convPSO: MTLComputePipelineState
    private let convWindowPSO: MTLComputePipelineState
    private let l2normPSO: MTLComputePipelineState
    private let gdnPSO: MTLComputePipelineState
    private let gatedNormPSO: MTLComputePipelineState
    private let attnEpiloguePSO: MTLComputePipelineState
    private let attnGatePSO: MTLComputePipelineState
    private let postAttnPSO: MTLComputePipelineState
    private let tailPSO: MTLComputePipelineState
    private let sharedGatePSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.convPSO = try context.pipeline("qwen_conv1d_silu")
        self.convWindowPSO = try context.pipeline("qwen_conv_window_update")
        self.l2normPSO = try context.pipeline("qwen_l2norm_qk")
        self.gdnPSO = try context.pipeline("qwen_gdn_recurse")
        self.gatedNormPSO = try context.pipeline("qwen_gdn_gated_norm")
        self.attnEpiloguePSO = try context.pipeline("qwen_attn_epilogue")
        self.attnGatePSO = try context.pipeline("qwen_attn_gate")
        self.postAttnPSO = try context.pipeline("qwen_post_attn_setup")
        self.tailPSO = try context.pipeline("qwen_layer_tail")
        self.sharedGatePSO = try context.pipeline("qwen_shared_gate_mul")
    }

    /// Depthwise causal conv + SiLU over the fused qkv channels.
    func encodeConv1dSilu(commandBuffer: MTLCommandBuffer,
                          qkv: MTLBuffer,
                          convWindow: MTLBuffer,
                          weight: MTLBuffer, weightOffset: Int,
                          convOut: MTLBuffer,
                          t: UInt32, channels: UInt32, kernelSize: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(convPSO)
        enc.setBuffer(qkv, offset: 0, index: 0)
        enc.setBuffer(convWindow, offset: 0, index: 1)
        enc.setBuffer(weight, offset: weightOffset, index: 2)
        enc.setBuffer(convOut, offset: 0, index: 3)
        var tt = t, cc = channels, kk = kernelSize
        enc.setBytes(&tt, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&cc, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&kk, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreads(MTLSize(width: Int(t), height: Int(channels), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 256, depth: 1))
        enc.endEncoding()
    }

    /// Roll the conv window to cover the next chunk.
    func encodeConvWindowUpdate(commandBuffer: MTLCommandBuffer,
                                qkv: MTLBuffer,
                                oldWindow: MTLBuffer,
                                newWindow: MTLBuffer,
                                t: UInt32, channels: UInt32, kernelSize: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(convWindowPSO)
        enc.setBuffer(qkv, offset: 0, index: 0)
        enc.setBuffer(oldWindow, offset: 0, index: 1)
        enc.setBuffer(newWindow, offset: 0, index: 2)
        var tt = t, cc = channels, kk = kernelSize
        enc.setBytes(&tt, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&cc, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&kk, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreads(MTLSize(width: Int(kernelSize - 1), height: Int(channels), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 1, height: 256, depth: 1))
        enc.endEncoding()
    }

    /// L2-normalize q/k heads in the conv output (in place).
    func encodeL2NormQK(commandBuffer: MTLCommandBuffer,
                        convOut: MTLBuffer,
                        t: UInt32, numHeads: UInt32, headDim: UInt32,
                        rowStride: UInt32,
                        eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(l2normPSO)
        enc.setBuffer(convOut, offset: 0, index: 0)
        var tt = t, hh = numHeads, dd = headDim, rr = rowStride, ee = eps
        enc.setBytes(&tt, length: MemoryLayout<UInt32>.size, index: 1)
        enc.setBytes(&hh, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&dd, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&rr, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&ee, length: MemoryLayout<Float>.size, index: 4)
        let width = min(Int(headDim), 256)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(numHeads * 2) * Int(t), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Gated delta rule recurrence over `t` tokens; fp32 state updated in
    /// place. One threadgroup per (v-head, dv row). `q`/`k`/`v` may alias one
    /// buffer via offsets (the conv output stores q|k|v contiguously).
    func encodeGDNRecurse(commandBuffer: MTLCommandBuffer,
                          q: MTLBuffer, qOffset: Int = 0,
                          k: MTLBuffer, kOffset: Int = 0,
                          v: MTLBuffer, vOffset: Int = 0,
                          a: MTLBuffer, aOffset: Int = 0,
                          b: MTLBuffer, bOffset: Int = 0,
                          aLog: MTLBuffer, aLogOffset: Int,
                          dtBias: MTLBuffer, dtBiasOffset: Int,
                          state: MTLBuffer,
                          y: MTLBuffer,
                          t: UInt32,
                          numKHeads: UInt32, numVHeads: UInt32,
                          keyHeadDim: UInt32, valueHeadDim: UInt32,
                          scalarStride: UInt32,
                          qkvStride: UInt32,
                          yStride: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(gdnPSO)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(v, offset: vOffset, index: 2)
        enc.setBuffer(a, offset: aOffset, index: 3)
        enc.setBuffer(b, offset: bOffset, index: 4)
        enc.setBuffer(aLog, offset: aLogOffset, index: 5)
        enc.setBuffer(dtBias, offset: dtBiasOffset, index: 6)
        enc.setBuffer(state, offset: 0, index: 7)
        enc.setBuffer(y, offset: 0, index: 8)
        var tt = t, hk = numKHeads, hv = numVHeads
        var dk = keyHeadDim, dv = valueHeadDim, st = scalarStride
        enc.setBytes(&tt, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&hk, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&hv, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&dk, length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBytes(&dv, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&st, length: MemoryLayout<UInt32>.size, index: 14)
        var qs = qkvStride, ys2 = yStride
        enc.setBytes(&qs, length: MemoryLayout<UInt32>.size, index: 15)
        enc.setBytes(&ys2, length: MemoryLayout<UInt32>.size, index: 16)
        if ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] != nil {
            print("[gdnbind] t=\(t) hk=\(hk) hv=\(hv) dk=\(dk) dv=\(dv) stride=\(st) "
                + "qOff=\(qOffset) kOff=\(kOffset) vOff=\(vOffset) aOff=\(aOffset) bOff=\(bOffset)")
        }
        enc.dispatchThreadgroups(
            MTLSize(width: Int(numVHeads * valueHeadDim), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// silu(z) * rms_norm(y, w) per (token, v-head).
    func encodeGDNGatedNorm(commandBuffer: MTLCommandBuffer,
                            y: MTLBuffer,
                            z: MTLBuffer,
                            weight: MTLBuffer, weightOffset: Int,
                            out: MTLBuffer,
                            t: UInt32, numVHeads: UInt32, valueHeadDim: UInt32,
                            eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(gatedNormPSO)
        enc.setBuffer(y, offset: 0, index: 0)
        enc.setBuffer(z, offset: 0, index: 1)
        enc.setBuffer(weight, offset: weightOffset, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        var tt = t, hv = numVHeads, dv = valueHeadDim, ee = eps
        enc.setBytes(&tt, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&hv, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dv, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&ee, length: MemoryLayout<Float>.size, index: 7)
        let width = min(Int(valueHeadDim), 256)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(numVHeads) * Int(t), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

        /// Per-head q/k RMSNorm + partial NeoX RoPE + sigmoid q gate. `qOut` may
    /// alias `q` for decode (single token).
    func encodeAttnEpilogue(commandBuffer: MTLCommandBuffer,
                            q: MTLBuffer, qOffset: Int,
                            qOut: MTLBuffer, qOutOffset: Int = 0,
                            qInStride: UInt32, qOutStride: UInt32,
                            k: MTLBuffer, kOffset: Int,
                            qWeight: MTLBuffer, qWeightOffset: Int,
                            kWeight: MTLBuffer, kWeightOffset: Int,
                            startPosition: UInt32,
                            t: UInt32,
                            headDim: UInt32,
                            numQHeads: UInt32, numKVHeads: UInt32,
                            rotatedPairs: UInt32,
                            theta: Float,
                            eps: Float) {
        precondition(headDim <= 256,
                     "qwen_attn_epilogue threadgroup memory is sized for headDim <= 256")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(attnEpiloguePSO)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(qOut, offset: qOutOffset, index: 12)
        enc.setBuffer(k, offset: kOffset, index: 1)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 2)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 3)
        var sp = startPosition, tt = t, hd = headDim
        var nq = numQHeads, nkv = numKVHeads, rp = rotatedPairs
        var th = theta, ee = eps
        var qis = qInStride, qos = qOutStride
        enc.setBytes(&sp, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&tt, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&nq, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&nkv, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&rp, length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&th, length: MemoryLayout<Float>.size, index: 10)
        enc.setBytes(&ee, length: MemoryLayout<Float>.size, index: 11)
        enc.setBytes(&qis, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&qos, length: MemoryLayout<UInt32>.size, index: 14)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(numQHeads + numKVHeads) * Int(t), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// attn *= sigmoid(gate) per token row (full-attention layers).
    func encodeAttnGate(commandBuffer: MTLCommandBuffer,
                        attn: MTLBuffer, attnOffset: Int,
                        gate: MTLBuffer, gateOffset: Int,
                        t: UInt32, d: UInt32,
                        headDim: UInt32,
                        gateStride: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(attnGatePSO)
        enc.setBuffer(attn, offset: attnOffset, index: 0)
        enc.setBuffer(gate, offset: gateOffset, index: 1)
        var tt = t, nh = d / headDim, hd = headDim, gs = gateStride
        enc.setBytes(&tt, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&nh, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&gs, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: Int(t), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// h += r in place, then normed2 = rms(h, w). One threadgroup per token.
    func encodePostAttnSetup(commandBuffer: MTLCommandBuffer,
                             h: MTLBuffer, hOffset: Int,
                             r: MTLBuffer, rOffset: Int,
                             normed2: MTLBuffer,
                             weight: MTLBuffer, weightOffset: Int,
                             t: UInt32, d: UInt32,
                             eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(postAttnPSO)
        enc.setBuffer(h, offset: hOffset, index: 0)
        enc.setBuffer(r, offset: rOffset, index: 1)
        enc.setBuffer(normed2, offset: 0, index: 2)
        enc.setBuffer(weight, offset: weightOffset, index: 3)
        var tt = t, dd = d, ee = eps
        enc.setBytes(&tt, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dd, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&ee, length: MemoryLayout<Float>.size, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: Int(t), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// hidden = h + shared + routed. One threadgroup per token.
    func encodeLayerTail(commandBuffer: MTLCommandBuffer,
                         hidden: MTLBuffer, hiddenOffset: Int,
                         h1: MTLBuffer, h1Offset: Int,
                         h2: MTLBuffer, h2Offset: Int,
                         t: UInt32, d: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(tailPSO)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
        enc.setBuffer(h1, offset: h1Offset, index: 1)
        enc.setBuffer(h2, offset: h2Offset, index: 2)
        var tt = t, dd = d
        enc.setBytes(&tt, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&dd, length: MemoryLayout<UInt32>.size, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: Int(t), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// sharedOut row t *= sigmoid(gateLogit[t]).
    func encodeSharedGateMul(commandBuffer: MTLCommandBuffer,
                             sharedOut: MTLBuffer,
                             gateLogit: MTLBuffer,
                             t: UInt32, d: UInt32) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(sharedGatePSO)
        enc.setBuffer(sharedOut, offset: 0, index: 0)
        enc.setBuffer(gateLogit, offset: 0, index: 1)
        var tt = t, dd = d
        enc.setBytes(&tt, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&dd, length: MemoryLayout<UInt32>.size, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: Int(t), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}