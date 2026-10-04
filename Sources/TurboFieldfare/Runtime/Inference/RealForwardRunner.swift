import Foundation
import Metal

public enum RDAdvicePolicyMode: String, Codable, Sendable, Equatable {
    case `default`
    case off
    case bounded
    case adaptive

    public static func parse(_ raw: String?) -> RDAdvicePolicyMode {
        switch raw?.lowercased() {
        case "off", "none", "disabled":
            return .off
        case "bounded":
            return .bounded
        case "adaptive":
            return .adaptive
        default:
            return .default
        }
    }
}

public struct RDAdviceAdaptivePolicyConfig: Sendable, Equatable {
    public var missCap: Int
    public var byteCap: UInt64
    public var slowCallNanos: UInt64

    public init(missCap: Int,
                byteCap: UInt64,
                slowCallNanos: UInt64) {
        self.missCap = missCap
        self.byteCap = byteCap
        self.slowCallNanos = slowCallNanos
    }

    public static let conservative = RDAdviceAdaptivePolicyConfig(
        missCap: 12,
        byteCap: 384 * 1_048_576,
        slowCallNanos: 1_000_000)
}

struct RDAdviceAdaptivePolicyState: Sendable, Equatable {
    var config: RDAdviceAdaptivePolicyConfig
    private var skipUntilPosition: Int = -1
    private(set) var recentSlowCallNanos: UInt64 = 0

    init(config: RDAdviceAdaptivePolicyConfig = .conservative) {
        self.config = config
    }

    mutating func reset() {
        skipUntilPosition = -1
        recentSlowCallNanos = 0
    }

    func shouldSkip(position: Int,
                    requestedMisses: Int,
                    estimatedBytes: UInt64,
                    canOverlapUsefulGPUWork: Bool) -> Bool {
        position <= skipUntilPosition ||
        !canOverlapUsefulGPUWork ||
        requestedMisses > config.missCap ||
        estimatedBytes > config.byteCap
    }

    mutating func update(after result: ExpertIOAdviceResult,
                                position: Int) {
        recentSlowCallNanos = max(recentSlowCallNanos, result.maxCallNanos)
        if result.maxCallNanos >= config.slowCallNanos {
            skipUntilPosition = max(skipUntilPosition, position)
        }
    }
}

/// Gemma 4 real-forward decode pass.
///
/// Composes the production kernels against the `.gturbo` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<30:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = (SWA) v_proj(a) | (full) k_proj(a)
///     per-head q/k_norm (bf16w), per-head v_norm (no_scale)
///     NeoX RoPE on Q + K (default for SWA, proportional for full)
///     write K and V into separate cache slots
///     attn = attention(scale=1.0, SWA window or full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)
///     h1 = rmsnorm_bf16w(h1, post_feedforward_layernorm_1)
///     // router + routed branch
///     xr   = rmsnorm_no_scale(h)
///     idx, w = router_topk_gemma4(xr, effective_scale[L], per_expert_scale[L])
///     h2 = rmsnorm_bf16w(h, pre_feedforward_layernorm_2)
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=0, routedBlobs=fetch(idx), w)
///     h2 = rmsnorm_bf16w(h2, post_feedforward_layernorm_2)
///     h = h + rmsnorm_bf16w(h1 + h2, post_feedforward_layernorm)
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), embed_table^T)
///   // final softcap and softmax happen in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
    case shared
    case routed
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    static func selectedDispatch(for family: PrefillProjectionFamily,
                                 chunkTokens: Int) -> PrefillProjectionDispatch {
        guard chunkTokens >= 32 else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return .repeatedGEMV
        case .kv, .o, .shared, .routed:
            return .qmm
        }
    }
}

public final class RealForwardRunner: ChunkedPrefillRunner, MultimodalPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, @unchecked Sendable {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Gemma-only sandwich norm on the shared branch output; nil for
        /// Qwen3.6 (whose shared branch feeds the tail unnormalized).
        let postF1: TensorView?
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let int8: DequantInt8GEMV
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue
    private let fusedPostAttentionSetup: FusedPostAttentionSetup
    private let fusedTail: FusedLayerTail
    private let laKernels: LinearAttnKernels

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillPostAttention: PrefillPostAttentionSetup
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillLayerTail: PrefillLayerTail
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    private let hidden: MTLBuffer        // [D] FP16
    private let normed: MTLBuffer        // [D] FP16
    private let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    private let kStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let vStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let oOut: MTLBuffer          // [D] FP16
    private let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    private let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: MTLBuffer // [F=2112] FP16
    private let denseScratchUp: MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: MTLBuffer  // [F=2112] FP16
    private let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    private let moeActs: MTLBuffer       // [topK * FmoE] FP16
    private let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    private let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    private let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    // Qwen3.6 decode scratch: fused qkv (q|k|v), z gate, a/b projections,
    // conv output, and the shared-expert gate logit.
    private let qkvBuf: MTLBuffer         // [convDim] FP16 (linear layers)
    private let zBuf: MTLBuffer           // [valueDim] FP16
    private let abBuf: MTLBuffer          // [2 * numVHeads] FP16 (a | b)
    private let convOutBuf: MTLBuffer     // [convDim] FP16 (post conv + silu)
    private let windowStageBuf: MTLBuffer // [kernel - 1, convDim] FP16 conv-window staging
    private let gateLogitBuf: MTLBuffer   // [1] FP16 shared-expert gate
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?

    private static let rdadviseBoundedMissCap = 12
    private static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    private static let rdadviseAdaptiveMissCap = 12
    private static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    private static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

/// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]
    /// Qwen3.6-only: all-ones per-expert scale fed to the top-k kernel so the
    /// renormalized softmax weights reach the experts unscaled.
    private let routerPerExpertOnes: MTLBuffer

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    private let useFusedGreedyHead: Bool
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    private var rdadviseSkipUntilPosition: Int = -1
    private var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    private var rdadviseAdaptivePosition: Int = -1
    private var rdadviseAdaptivePositionBytes: UInt64 = 0
    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        self.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        self.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: useFP16Ring,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: max(
                                        PrefillRuntimeConfig.maxChunkTokens,
                                        VisionConfig().maximumPooledTokens))

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(context: context)
        self.int8      = try DequantInt8GEMV(context: context)
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits,
                                                  activation: GatedActivation(
                                                    hiddenActivation: cfg.hiddenActivation))
        self.moe       = try MoE(context: context,
                                 activation: GatedActivation(
                                    hiddenActivation: cfg.hiddenActivation))
        self.laKernels = try LinearAttnKernels(context: context)
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.fusedPostAttentionSetup = try FusedPostAttentionSetup(context: context)
        self.fusedTail = try FusedLayerTail(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(context: context)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillPostAttention = try PrefillPostAttentionSetup(context: context)
        self.prefillRouter = try PrefillRouter(context: context)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.sharedExpertWeightBits,
            activation: GatedActivation(hiddenActivation: cfg.hiddenActivation))
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(
            context: context,
            activation: GatedActivation(hiddenActivation: cfg.hiddenActivation))
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context,
                                                               maxD: cfg.hiddenSize)

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return b
        }
        self.hidden        = try buf(D)
        self.normed        = try buf(D)
        self.attnOut       = try buf(maxQ)
        self.qScratch      = try buf(maxQ)
        self.kStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.vStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.oOut          = try buf(D)
        self.h1Buf         = try buf(D)
        self.h2Buf         = try buf(D)
        self.routedX       = try buf(D)
        self.denseX        = try buf(D)
        self.denseScratchGate = try buf(F)
        self.denseScratchUp   = try buf(F)
        self.denseScratchAct  = try buf(F)
        self.routerInput   = try buf(D)
        self.zeroResidual  = try buf(D)
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices    = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.outWeights    = try buf(cfg.topKExperts)
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize)
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.greedyTokenBuf = tok

        // Qwen3.6 linear-attention scratch (convDim = 2*keyDim + valueDim).
        let convDim = cfg.linearNumKeyHeads * cfg.linearKeyHeadDim * 2
            + cfg.linearNumValueHeads * cfg.linearValueHeadDim
        self.qkvBuf     = try buf(max(convDim, 1))
        self.convOutBuf = try buf(max(convDim, 1))
        self.windowStageBuf = try buf(max((cfg.linearConvKernelDim - 1) * convDim, 1))
        self.zBuf       = try buf(max(cfg.linearNumValueHeads * cfg.linearValueHeadDim, 1))
        self.abBuf      = try buf(max(cfg.linearNumValueHeads * 2, 1))
        self.gateLogitBuf = try buf(1)

        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: rows,
                                 cols: cols)
        }
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate, rows: UInt32(F), cols: UInt32(D)),
                up: sharedProj(up, rows: UInt32(F), cols: UInt32(D)),
                down: sharedProj(down, rows: UInt32(D), cols: UInt32(F)),
                postF1: cfg.usesLinearAttention ? nil : (try model.postFFN1(layer: L))))
        }
        self.sharedExpertProjections = sharedViews

        // Pre-fold 1/sqrt(D) into router.scale per layer. Each layer gets its
        // own BF16 [D] buffer — the kernel reads `effective_scale[i]` and we
        // pay for the multiply once per generation, not per token. Qwen3.6 has
        // no router scale at all, so every layer shares an all-ones buffer.
        var perLayer: [MTLBuffer] = []
        perLayer.reserveCapacity(cfg.numLayers)
        let invSqrtD = Float(1.0) / Float(D).squareRoot()
        let dInts = D
        if cfg.usesLinearAttention {
            guard let ones = device.makeBuffer(
                length: dInts * MemoryLayout<UInt16>.size,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let dst = ones.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<dInts { dst[i] = Quantization.bf16Bits(1.0) }
            ones.label = "router.ones_scale"
            for _ in 0..<cfg.numLayers { perLayer.append(ones) }
            // Qwen3.6's router applies no per-expert scale; the top-k kernel
            // multiplies one anyway, so hand it an all-ones vector too.
            guard let expertOnes = device.makeBuffer(
                length: cfg.numExperts * MemoryLayout<UInt16>.size,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let edst = expertOnes.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<cfg.numExperts { edst[i] = Quantization.bf16Bits(1.0) }
            expertOnes.label = "router.ones_per_expert"
            self.routerPerExpertOnes = expertOnes
        } else {
            self.routerPerExpertOnes = try buf(cfg.numExperts)
        }
        for L in 0..<cfg.numLayers where !cfg.usesLinearAttention {
            let scaleView = try model.routerScale(layer: L)
            guard let buf = device.makeBuffer(length: dInts * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let src = scaleView.buffer.contents()
                .advanced(by: Int(scaleView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<dInts {
                let v = Quantization.bf16ToFloat(src[i]) * invSqrtD
                dst[i] = Quantization.bf16Bits(v)
            }
            buf.label = "effective_scale.L\(L)"
            perLayer.append(buf)
        }
        self.effectiveScaleBuffers = perLayer
    }

    public func reset() {
        kv?.reset()
        resetTransientState()
    }

    /// Abandons the newest KV tokens so a stateful conversation can resume at
    /// an earlier visible boundary.
    public func rewind(to position: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "rewind requires an initialized KV cache")
        }
        guard position > 0, position <= kv.position,
              kv.highWaterPosition - position <= kv.maxRewindTokens else {
            throw PrefillError.prefillCursorMismatch(
                "cannot rewind KV from \(kv.position) to \(position); "
                    + "high water \(kv.highWaterPosition), "
                    + "ring slack \(kv.maxRewindTokens)")
        }
        kv.rewind(to: position)
        resetTransientState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        resetTransientState()
    }

    private func resetTransientState() {
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
    }

    public private(set) var totalIoNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    public private(set) var totalRDAdviseNanos: UInt64 = 0
    public private(set) var totalRDAdviseCalls: UInt64 = 0
    public private(set) var totalRDAdviseBytes: UInt64 = 0
    public private(set) var totalRDAdviseFailures: UInt64 = 0
    public private(set) var totalRDAdviseSkipped: UInt64 = 0

    private func recordRDAdvice(_ result: ExpertIOAdviceResult, wallNanos: UInt64) {
        totalRDAdviseNanos &+= wallNanos
        totalRDAdviseCalls &+= UInt64(result.calls)
        totalRDAdviseBytes &+= result.bytes
        totalRDAdviseFailures &+= UInt64(result.failed)
        totalRDAdviseSkipped &+= UInt64(result.skipped)
    }

    private func shouldSkipRDAdvice(position: Int,
                                    requestedMisses: Int,
                                    estimatedBytes: UInt64,
                                    canOverlapUsefulGPUWork: Bool) -> ExpertIOAdviceResult? {
        switch rdadvisePolicyMode {
        case .bounded:
            if position <= rdadviseSkipUntilPosition {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            if requestedMisses > Self.rdadviseBoundedMissCap {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            return nil
        case .adaptive:
            if position != rdadviseAdaptivePosition {
                rdadviseAdaptivePosition = position
                rdadviseAdaptivePositionBytes = 0
            }
            let cumulativeEstimatedBytes = rdadviseAdaptivePositionBytes &+ estimatedBytes
            let shouldSkip = rdadviseAdaptiveState.shouldSkip(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: cumulativeEstimatedBytes,
                canOverlapUsefulGPUWork: canOverlapUsefulGPUWork)
            rdadviseAdaptivePositionBytes = cumulativeEstimatedBytes
            guard shouldSkip else { return nil }
            return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                bytes: estimatedBytes)
        case .default, .off:
            return nil
        }
    }

    private func updateRDAdvicePolicy(after result: ExpertIOAdviceResult,
                                      position: Int) {
        switch rdadvisePolicyMode {
        case .bounded:
            if result.maxCallNanos > Self.rdadviseBoundedMaxCallNanos {
                rdadviseSkipUntilPosition = max(rdadviseSkipUntilPosition, position + 1)
            }
        case .adaptive:
            rdadviseAdaptiveState.update(after: result, position: position)
        case .default, .off:
            break
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        try await PrefillSpanIteration.forEachSpan(spans) { spanIndex, span in
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: config,
                writeFinalHead: spanIndex == spans.count - 1)
            onProgress(span.completedCount)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }


    public func prefillMultimodal(input: MultimodalPrefillInput,
                                  startPosition: Int,
                                  outputMode: PrefillOutputMode,
                                  config: PrefillRuntimeConfig,
                                  into logits: MTLBuffer,
                                  onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillMultimodal")
        // The same property `coercedForImagePrompt()` repairs, so the guard and
        // the coercion cannot drift apart.
        guard config.servesImagePrompt else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill requires the complete chunked prefill path")
        }
        let tokens = input.embeddingTokenIDs
        guard startPosition >= 0,
              startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill range exceeds maxContext \(maxContext)")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        // The planner applies the same clamp the scratch layout does. Cutting at
        // the raw `config.chunkTokens` let a caller size a 280-token scratch and
        // then hand it a larger chunk, which died on the guard.
        let textChunkTokens = min(config.chunkTokens, PrefillRuntimeConfig.maxChunkTokens)
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: tokens.count,
            imageRanges: input.imageSpans.map(\.tokenRange),
            chunkTokens: config.chunkTokens)
        let imageBuffers = input.imageSpans.map(\.features.buffer)

        // One layout for the whole turn, sized for the largest item in it.
        // Keying the single-slot cache on each item's exact chunk size made a
        // text-image-text prompt free and reallocate the entire chunk scratch at
        // every boundary, twice per image span, every turn.
        let pooledTokens = VisionConfig().maximumPooledTokens
        let hasImages = work.contains(where: \.isImage)
        let layoutTokens = hasImages
            ? max(config.chunkTokens, pooledTokens)
            : config.chunkTokens
        let layoutLimit = hasImages
            ? max(PrefillRuntimeConfig.maxChunkTokens, pooledTokens)
            : PrefillRuntimeConfig.maxChunkTokens
        let scratch = try ensurePrefillScratch(
            config: config.replacingChunkTokens(layoutTokens),
            chunkTokenLimit: layoutLimit)

        var completed = 0
        for (index, item) in work.enumerated() {
            try Task.checkCancellation()
            let isImage = item.isImage
            let imageFeatures = item.imageIndex.map { imageBuffers[$0] }
            // Every item executes under the chunk size it was planned at.
            let chunkConfig = config.replacingChunkTokens(
                isImage ? item.range.count : textChunkTokens)
            let lower = tokens.index(tokens.startIndex, offsetBy: item.range.lowerBound)
            let upper = tokens.index(tokens.startIndex, offsetBy: item.range.upperBound)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: startPosition + item.range.lowerBound,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: chunkConfig,
                writeFinalHead: index == work.count - 1,
                embeddingOverride: imageFeatures,
                bidirectionalBlock: isImage
                    ? (startPosition + item.range.lowerBound)..<(startPosition + item.range.upperBound)
                    : nil)
            completed += item.range.count
            onProgress(completed)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    @discardableResult
    private func ensurePrefillScratch(
        config: PrefillRuntimeConfig,
        chunkTokenLimit: Int = PrefillRuntimeConfig.maxChunkTokens
    ) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg,
                                               chunkTokens: config.chunkTokens,
                                               chunkTokenLimit: chunkTokenLimit)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool,
                                     embeddingOverride: MTLBuffer? = nil,
                                     bidirectionalBlock: Range<Int>? = nil) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires FP16 KV")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }

        struct LayerPrefillQKVViews {
            let inputNorm: TensorView
            let q: TensorView
            let k: TensorView
            let v: TensorView
            let o: TensorView
            let postAttention: TensorView
            let preFFN: TensorView
            let preFFN2: TensorView
            let postFFN2: TensorView
            let postFFN: TensorView
            let layerScalar: TensorView
            let qNorm: TensorView
            let kNorm: TensorView
            let router: TensorView
            let routerPerExpertScale: TensorView
        }

        let layerViews: [LayerPrefillQKVViews] = cfg.usesLinearAttention
            ? []
            : try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                q: try model.qProj(layer: L),
                k: try model.kProj(layer: L),
                v: isFull ? (try model.kProj(layer: L)) : (try model.vProj(layer: L)),
                o: try model.oProj(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                preFFN: try model.preFFN(layer: L),
                preFFN2: try model.preFFN2(layer: L),
                postFFN2: try model.postFFN2(layer: L),
                postFFN: try model.postFFN(layer: L),
                layerScalar: try model.layerScalar(layer: L),
                qNorm: try model.qNorm(layer: L),
                kNorm: try model.kNorm(layer: L),
                router: try model.router(layer: L),
                routerPerExpertScale: try model.routerPerExpertScale(layer: L))
        }

        // Qwen3.6 views: full-attention tensors on the mask's 1-layers, the
        // gated-delta-rule tensor set on the rest, plus the scalar
        // shared-expert gate.
        struct QwenLayerPrefillViews {
            let inputNorm: TensorView
            let postAttention: TensorView
            let router: TensorView
            let sharedGate: TensorView
            let q: TensorView?
            let k: TensorView?
            let v: TensorView?
            let o: TensorView?
            let qNorm: TensorView?
            let kNorm: TensorView?
            let inProjQKV: TensorView?
            let inProjZ: TensorView?
            let inProjA: TensorView?
            let inProjB: TensorView?
            let conv1d: TensorView?
            let aLog: TensorView?
            let dtBias: TensorView?
            let outProj: TensorView?
            let normW: TensorView?
        }
        let qwenViews: [QwenLayerPrefillViews] = cfg.usesLinearAttention
            ? try (0..<cfg.numLayers).map { L in
                let isFull = cfg.fullAttentionLayerMask[L] != 0
                return QwenLayerPrefillViews(
                    inputNorm: try model.inputNorm(layer: L),
                    postAttention: try model.postAttnNorm(layer: L),
                    router: try model.router(layer: L),
                    sharedGate: try model.sharedExpertScalarGate(layer: L),
                    q: isFull ? try model.qProj(layer: L) : nil,
                    k: isFull ? try model.kProj(layer: L) : nil,
                    v: isFull ? try model.vProj(layer: L) : nil,
                    o: isFull ? try model.oProj(layer: L) : nil,
                    qNorm: isFull ? try model.qNorm(layer: L) : nil,
                    kNorm: isFull ? try model.kNorm(layer: L) : nil,
                    inProjQKV: isFull ? nil : try model.linearInProjQKV(layer: L),
                    inProjZ: isFull ? nil : try model.linearInProjZ(layer: L),
                    inProjA: isFull ? nil : try model.linearInProjA(layer: L),
                    inProjB: isFull ? nil : try model.linearInProjB(layer: L),
                    conv1d: isFull ? nil : try model.linearConv1d(layer: L),
                    aLog: isFull ? nil : try model.linearALog(layer: L),
                    dtBias: isFull ? nil : try model.linearDtBias(layer: L),
                    outProj: isFull ? nil : try model.linearOutProj(layer: L),
                    normW: isFull ? nil : try model.linearNorm(layer: L))
            }
            : []

        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = ctx.device.makeBuffer(bytes: tokenIDs,
                                                      length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let sqrtHidden = cfg.embedOutScale
        let t = tokens.count
        let emb = model.embedding

        func encodeInt4Projection(commandBuffer: MTLCommandBuffer,
                                  family: PrefillProjectionFamily,
                                  weights: TensorView,
                                  x: MTLBuffer,
                                  y: MTLBuffer,
                                  rows: Int,
                                  columns: Int,
                                  tokenCount: Int,
                                  xStrideElements: Int,
                                  yStrideElements: Int,
                                  yBaseElements: Int = 0) {
            if tokenCount >= 32,
               family == .q || family == .kv || family == .o,
               let candidate = prefillMPPAffineInt4 {
                let metadata = candidate.encode(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    yOffset: yBaseElements * MemoryLayout<Float16>.stride,
                    m: tokenCount,
                    n: rows,
                    k: columns)
                if metadata.path == .affineThreadgroupF16 {
                    return
                }
            }
            if PrefillProjectionDispatchPolicy.selectedDispatch(for: family,
                                                                chunkTokens: tokenCount) == .qmm {
                prefillQMM.encode(commandBuffer: commandBuffer,
                                  weights: weights.buffer,
                                  weightsOffset: Int(weights.offset),
                                  scales: weights.buffer,
                                  scalesOffset: Int(weights.scaleOffset),
                                  biases: weights.buffer,
                                  biasesOffset: Int(weights.biasOffset),
                                  x: x,
                                  y: y,
                                  yOffset: yBaseElements * MemoryLayout<Float16>.stride,
                                  t: tokenCount,
                                  n: rows,
                                  k: columns)
                return
            }
            for row in 0..<tokenCount {
                int4.encode(commandBuffer: commandBuffer,
                            weights: weights.buffer,
                            weightsOffset: Int(weights.offset),
                            scales: weights.buffer,
                            scalesOffset: Int(weights.scaleOffset),
                            biases: weights.buffer,
                            biasesOffset: Int(weights.biasOffset),
                            x: x,
                            xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                            y: y,
                            yOffset: (yBaseElements
                                      + row * yStrideElements) * MemoryLayout<Float16>.stride,
                            m: UInt32(rows),
                            n: UInt32(columns))
            }
        }

        func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                           source: MTLBuffer,
                           destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                           sourceTokenOffset: Int,
                           tokenCount: Int,
                           bytesPerToken: Int) throws {
            guard tokenCount > 0 else { return }
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(from: source,
                      sourceOffset: sourceTokenOffset * bytesPerToken,
                      to: destination.buffer,
                      destinationOffset: destination.offset,
                      size: tokenCount * bytesPerToken)
            blit.endEncoding()
        }

        func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                                  kv: KVCacheManager,
                                  layer: Int,
                                  startPosition: Int,
                                  tokenCount: Int,
                                  keySource: MTLBuffer,
                                  valueSource: MTLBuffer,
                                  bytesPerToken: Int) throws {
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
            let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keyFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            guard firstSpan < tokenCount else { return }

            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
            let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keySecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueSecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
        }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        cb.label = "prefill start=\(startPosition) count=\(t) phase=embed"
        if let embeddingOverride {
            // The override spans the whole chunk — the planner emits image spans
            // as standalone chunks — so the INT4 gather it would cover is
            // skipped rather than encoded and fully overwritten.
            let bytes = t * D * MemoryLayout<Float16>.stride
            guard embeddingOverride.length >= bytes,
                  let blit = cb.makeBlitCommandEncoder() else {
                throw VisionRuntimeError.invalidInput(
                    "projected image feature buffer is too small")
            }
            blit.copy(from: embeddingOverride,
                      sourceOffset: 0,
                      to: scratch.hidden,
                      destinationOffset: 0,
                      size: bytes)
            blit.endEncoding()
        } else {
        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer,
                            tableOffset: Int(emb.offset),
                            scales: emb.buffer,
                            scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer,
                            biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: scratch.hidden,
                            t: UInt32(t),
                            d: UInt32(D),
                            outScale: sqrtHidden)
        }
        if let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
            Self.debugDumpBuffer(scratch.hidden,
                                 elements: t * D,
                                 to: "\(dumpDir)/embed.f16",
                                 commandBuffer: cb)
        }

        for L in 0..<cfg.numLayers {
            model.beginOpeningRoutedExpertStreamer(layer: L)
            if cfg.usesLinearAttention {
                let v = qwenViews[L]
                let isFullL = cfg.fullAttentionLayerMask[L] != 0
                let convDim = cfg.linearNumKeyHeads * cfg.linearKeyHeadDim * 2
                    + cfg.linearNumValueHeads * cfg.linearValueHeadDim
                let keyDim = cfg.linearNumKeyHeads * cfg.linearKeyHeadDim
                let valueDim = cfg.linearNumValueHeads * cfg.linearValueHeadDim
                let qDimQwen = cfg.numHeads * cfg.headDim
                let kvDimQwen = cfg.numFullKVHeads * cfg.fullHeadDim
                let nvQwen = cfg.linearNumValueHeads

                prefillRMS.encodeBF16W(commandBuffer: cb,
                                       x: scratch.hidden,
                                       weight: v.inputNorm.buffer,
                                       weightOffset: Int(v.inputNorm.offset),
                                       out: scratch.normed,
                                       t: UInt32(t),
                                       d: UInt32(D),
                                       eps: eps)
                if isFullL {
                    let q = v.q!, k = v.k!, vv = v.v!, o = v.o!
                    let qN = v.qNorm!, kN = v.kNorm!
                    encodeInt4Projection(commandBuffer: cb, family: .q,
                                         weights: q, x: scratch.normed,
                                         y: scratch.qwenQKV,
                                         rows: qDimQwen * 2, columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: qDimQwen * 2)
                    encodeInt4Projection(commandBuffer: cb, family: .kv,
                                         weights: k, x: scratch.normed,
                                         y: scratch.kStage,
                                         rows: kvDimQwen, columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: kvDimQwen)
                    encodeInt4Projection(commandBuffer: cb, family: .kv,
                                         weights: vv, x: scratch.normed,
                                         y: scratch.vStage,
                                         rows: kvDimQwen, columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: kvDimQwen)
                    laKernels.encodeAttnEpilogue(
                        commandBuffer: cb,
                        q: scratch.qwenQKV, qOffset: 0,
                        qOut: scratch.q,
                        qInStride: UInt32(qDimQwen * 2),
                        qOutStride: UInt32(qDimQwen),
                        k: scratch.kStage, kOffset: 0,
                        qWeight: qN.buffer, qWeightOffset: Int(qN.offset),
                        kWeight: kN.buffer, kWeightOffset: Int(kN.offset),
                        startPosition: UInt32(startPosition),
                        t: UInt32(t),
                        headDim: UInt32(cfg.headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(cfg.numFullKVHeads),
                        rotatedPairs: UInt32(
                            Double(cfg.headDim) * cfg.partialRotaryFactor / 2.0),
                        theta: Float(cfg.fullRopeTheta),
                        eps: eps)
                    if let kv {
                        let bytes = kvDimQwen * MemoryLayout<Float16>.stride
                        try copyPrefillKVToCache(commandBuffer: cb,
                                                 kv: kv,
                                                 layer: L,
                                                 startPosition: startPosition,
                                                 tokenCount: t,
                                                 keySource: scratch.kStage,
                                                 valueSource: scratch.vStage,
                                                 bytesPerToken: bytes)
                    }
                    let params = PrefillAttentionParams(
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(cfg.headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(cfg.numFullKVHeads),
                        kvValidCount: UInt32(startPosition + t),
                        slidingWindow: UInt32(startPosition + t),
                        kvTokenStrideElements: UInt32(kvDimQwen),
                        qTokenStrideElements: UInt32(qDimQwen),
                        oTokenStrideElements: UInt32(qDimQwen),
                        scale: cfg.attentionScale,
                        bidirectionalBlockStart: 0,
                        bidirectionalBlockEnd: 0)
                    if let kv {
                        let keyBuffer = kv.keyBuffer(layer: L,
                                                     validTokenCount: startPosition + t)
                        let valueBuffer = kv.valueBuffer(layer: L,
                                                         validTokenCount: startPosition + t)
                        prefillAttention.encodeCausal(
                            commandBuffer: cb,
                            q: scratch.q,
                            k: keyBuffer,
                            v: valueBuffer,
                            out: scratch.attentionOutput,
                            params: params,
                            kvRingCapacity: 0,
                            layerKind: .full,
                            path: prefillAttentionPath)
                    } else {
                        throw PrefillError.chunkedUnsupported(
                            "chunked prefill attention requires FP16 KV")
                    }
                    if L == 3, let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                        Self.debugDumpBuffer(scratch.attentionOutput,
                                             elements: t * qDimQwen,
                                             to: "\(dumpDir)/l3_att_ungated.f16",
                                             commandBuffer: cb)
                    }
                    laKernels.encodeAttnGate(
                        commandBuffer: cb,
                        attn: scratch.attentionOutput, attnOffset: 0,
                        gate: scratch.qwenQKV,
                        gateOffset: 0,
                        t: UInt32(t), d: UInt32(qDimQwen),
                        headDim: UInt32(cfg.headDim),
                        gateStride: UInt32(convDim))
                    encodeInt4Projection(commandBuffer: cb, family: .o,
                                         weights: o, x: scratch.attentionOutput,
                                         y: scratch.h1,
                                         rows: D, columns: qDimQwen,
                                         tokenCount: t,
                                         xStrideElements: qDimQwen,
                                         yStrideElements: D)
                } else {
                    let stateBuf = try kv?.stateBuffer(layer: L)
                    let convWindow = try kv?.convWindowBuffer(layer: L)
                    guard let stateBuf, let convWindow else {
                        throw PrefillError.chunkedUnsupported(
                            "linear layer \(L) is missing its state buffers")
                    }
                    encodeInt4Projection(commandBuffer: cb, family: .q,
                                         weights: v.inProjQKV!,
                                         x: scratch.normed,
                                         y: scratch.qwenQKV,
                                         rows: convDim, columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: convDim)
                    encodeInt4Projection(commandBuffer: cb, family: .kv,
                                         weights: v.inProjZ!,
                                         x: scratch.normed,
                                         y: scratch.qwenZ,
                                         rows: valueDim, columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: valueDim)
                    encodeInt4Projection(commandBuffer: cb, family: .kv,
                                         weights: v.inProjA!,
                                         x: scratch.normed,
                                         y: scratch.qwenAB,
                                         rows: nvQwen, columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: nvQwen * 2)
                    encodeInt4Projection(commandBuffer: cb, family: .kv,
                                         weights: v.inProjB!,
                                         x: scratch.normed,
                                         y: scratch.qwenAB,
                                         rows: nvQwen, columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: nvQwen * 2,
                                         yBaseElements: nvQwen)
                    laKernels.encodeConv1dSilu(
                        commandBuffer: cb,
                        qkv: scratch.qwenQKV,
                        convWindow: convWindow,
                        weight: v.conv1d!.buffer,
                        weightOffset: Int(v.conv1d!.offset),
                        convOut: scratch.qwenConvOut,
                        t: UInt32(t),
                        channels: UInt32(convDim),
                        kernelSize: UInt32(cfg.linearConvKernelDim))
                    laKernels.encodeConvWindowUpdate(
                        commandBuffer: cb,
                        qkv: scratch.qwenQKV,
                        oldWindow: convWindow,
                        newWindow: scratch.qwenConvWindowStage,
                        t: UInt32(t),
                        channels: UInt32(convDim),
                        kernelSize: UInt32(cfg.linearConvKernelDim))
                    if let blit = cb.makeBlitCommandEncoder() {
                        blit.copy(
                            from: scratch.qwenConvWindowStage, sourceOffset: 0,
                            to: convWindow, destinationOffset: 0,
                            size: (cfg.linearConvKernelDim - 1) * convDim
                                * MemoryLayout<Float16>.stride)
                        blit.endEncoding()
                    }
                    laKernels.encodeL2NormQK(
                        commandBuffer: cb,
                        convOut: scratch.qwenConvOut,
                        t: UInt32(t),
                        numHeads: UInt32(cfg.linearNumKeyHeads),
                        headDim: UInt32(cfg.linearKeyHeadDim),
                        rowStride: UInt32(convDim),
                        eps: eps)
                    if L == 0, let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                        Self.debugDumpBuffer(scratch.qwenConvOut, elements: t * convDim,
                                             to: "\(dumpDir)/l0_conv_l2.f16", commandBuffer: cb)
                        Self.debugDumpBuffer(scratch.qwenZ, elements: t * valueDim,
                                             to: "\(dumpDir)/l0_z.f16", commandBuffer: cb)
                        Self.debugDumpBuffer(scratch.qwenAB, elements: t * nvQwen * 2,
                                             to: "\(dumpDir)/l0_ab.f16", commandBuffer: cb)
                    }
                    laKernels.encodeGDNRecurse(
                        commandBuffer: cb,
                        q: scratch.qwenConvOut, qOffset: 0,
                        k: scratch.qwenConvOut,
                        kOffset: keyDim * MemoryLayout<Float16>.stride,
                        v: scratch.qwenConvOut,
                        vOffset: keyDim * 2 * MemoryLayout<Float16>.stride,
                        a: scratch.qwenAB, aOffset: 0,
                        b: scratch.qwenAB,
                        bOffset: nvQwen * MemoryLayout<Float16>.stride,
                        aLog: v.aLog!.buffer, aLogOffset: Int(v.aLog!.offset),
                        dtBias: v.dtBias!.buffer,
                        dtBiasOffset: Int(v.dtBias!.offset),
                        state: stateBuf,
                        y: scratch.q,
                        t: UInt32(t),
                        numKHeads: UInt32(cfg.linearNumKeyHeads),
                        numVHeads: UInt32(nvQwen),
                        keyHeadDim: UInt32(cfg.linearKeyHeadDim),
                        valueHeadDim: UInt32(cfg.linearValueHeadDim),
                        scalarStride: UInt32(nvQwen * 2),
                        qkvStride: UInt32(convDim),
                        yStride: UInt32(valueDim))
                    laKernels.encodeGDNGatedNorm(
                        commandBuffer: cb,
                        y: scratch.q,
                        z: scratch.qwenZ,
                        weight: v.normW!.buffer,
                        weightOffset: Int(v.normW!.offset),
                        out: scratch.q,
                        t: UInt32(t),
                        numVHeads: UInt32(nvQwen),
                        valueHeadDim: UInt32(cfg.linearValueHeadDim),
                        eps: eps)
                    encodeInt4Projection(commandBuffer: cb, family: .o,
                                         weights: v.outProj!,
                                         x: scratch.q,
                                         y: scratch.h1,
                                         rows: D, columns: valueDim,
                                         tokenCount: t,
                                         xStrideElements: valueDim,
                                         yStrideElements: D)
                }

                if L == 3, let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                    Self.debugDumpBuffer(scratch.attentionOutput,
                                         elements: t * qDimQwen,
                                         to: "\(dumpDir)/l3_attn.f16", commandBuffer: cb)
                    if let kv {
                        Self.debugDumpBuffer(kv.keyBuffer(layer: L, validTokenCount: startPosition + t),
                                             elements: (startPosition + t) * kvDimQwen,
                                             to: "\(dumpDir)/l3_cacheK.f16", commandBuffer: cb)
                        Self.debugDumpBuffer(kv.valueBuffer(layer: L, validTokenCount: startPosition + t),
                                             elements: (startPosition + t) * kvDimQwen,
                                             to: "\(dumpDir)/l3_cacheV.f16", commandBuffer: cb)
                    }
                    Self.debugDumpBuffer(scratch.h1, elements: t * D,
                                         to: "\(dumpDir)/l3_outproj.f16", commandBuffer: cb)
                    Self.debugDumpBuffer(scratch.q, elements: t * qDimQwen,
                                         to: "\(dumpDir)/l3_q.f16", commandBuffer: cb)
                }
                laKernels.encodePostAttnSetup(
                    commandBuffer: cb,
                    h: scratch.hidden, hOffset: 0,
                    r: scratch.h1, rOffset: 0,
                    normed2: scratch.qwenNormed2,
                    weight: v.postAttention.buffer,
                    weightOffset: Int(v.postAttention.offset),
                    t: UInt32(t), d: UInt32(D),
                    eps: eps)
                if L == 3, let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                    Self.debugDumpBuffer(scratch.qwenNormed2, elements: t * D,
                                         to: "\(dumpDir)/l3_normed2.f16", commandBuffer: cb)
                }
                if L == 0, let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                    Self.debugDumpBuffer(scratch.normed, elements: t * D,
                                         to: "\(dumpDir)/l0_normed.f16", commandBuffer: cb)
                    Self.debugDumpBuffer(scratch.qwenQKV, elements: t * 8192,
                                         to: "\(dumpDir)/l0_qkv.f16", commandBuffer: cb)
                    Self.debugDumpBuffer(scratch.qwenConvOut, elements: t * 8192,
                                         to: "\(dumpDir)/l0_conv.f16", commandBuffer: cb)
                    Self.debugDumpBuffer(scratch.q, elements: t * 4096,
                                         to: "\(dumpDir)/l0_gdn.f16", commandBuffer: cb)
                    Self.debugDumpBuffer(scratch.qwenAB, elements: t * 64,
                                         to: "\(dumpDir)/l0_ab.f16", commandBuffer: cb)

                    Self.debugDumpBuffer(scratch.h1, elements: t * D,
                                         to: "\(dumpDir)/l0_outproj.f16", commandBuffer: cb)
                    Self.debugDumpBuffer(scratch.qwenNormed2, elements: t * D,
                                         to: "\(dumpDir)/l0_normed2.f16", commandBuffer: cb)
                }
                prefillRouter.encodeGemma4Block(
                    commandBuffer: cb,
                    weights: v.router.buffer,
                    weightsOffset: Int(v.router.offset),
                    scales: v.router.buffer,
                    scalesOffset: Int(v.router.scaleOffset),
                    biases: v.router.buffer,
                    biasesOffset: Int(v.router.biasOffset),
                    hidden: scratch.qwenNormed2,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: routerPerExpertOnes,
                    outIndices: scratch.routeIDs,
                    outWeights: scratch.routeWeights,
                    queryCount: UInt32(t),
                    numExperts: UInt32(cfg.numExperts),
                    d: UInt32(D),
                    topK: UInt32(cfg.topKExperts),
                    hiddenStrideElements: UInt32(D))
                cb.commit()
                try waitForCompletion(cb)

                let routeCount = t * cfg.topKExperts
                let idPtr = scratch.routeIDs.contents()
                    .bindMemory(to: UInt32.self, capacity: routeCount)
                let weightPtr = scratch.routeWeights.contents()
                    .bindMemory(to: Float16.self, capacity: routeCount)
                var routeIDs = [UInt32]()
                routeIDs.reserveCapacity(routeCount)
                var routeWeights = [Float16]()
                routeWeights.reserveCapacity(routeCount)
                for i in 0..<routeCount {
                    routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                    routeWeights.append(weightPtr[i])
                }
                if L == 0, let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                    let idData = routeIDs.withUnsafeBufferPointer { ptr in
                        Data(bytes: ptr.baseAddress!, count: routeCount * 4)
                    }
                    try? idData.write(to: URL(fileURLWithPath: "\(dumpDir)/l0_routeids.u32"))
                    let wData = routeWeights.withUnsafeBufferPointer { ptr in
                        Data(bytes: ptr.baseAddress!, count: routeCount * 2)
                    }
                    try? wData.write(to: URL(fileURLWithPath: "\(dumpDir)/l0_routew.f16"))
                }
                let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                               weights: routeWeights,
                                                               queryCount: t,
                                                               topK: cfg.topKExperts)
                let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                let routeTileExpertCount: Int
                if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
                    guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                        throw PrefillError.chunkedUnsupported(
                            "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
                    }
                    routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
                } else {
                    routeTileExpertCount = schedulerConfig.tileExperts
                }
                let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                    pairs,
                    queryCount: t,
                    topK: cfg.topKExperts,
                    numExperts: cfg.numExperts,
                    tileExpertCount: routeTileExpertCount,
                    expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

                guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                sharedCB.label =
                    "prefill start=\(startPosition) count=\(t) layer=\(L) phase=qwen_shared"
                let sharedProj = sharedExpertProjections[L]
                try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                    x: scratch.qwenNormed2,
                                                    y: scratch.h1,
                                                    gate: sharedProj.gate,
                                                    up: sharedProj.up,
                                                    down: sharedProj.down,
                                                    scratchGate: scratch.sharedGateScratch,
                                                    scratchUp: scratch.sharedUpScratch,
                                                    scratchAct: scratch.sharedActScratch,
                                                    queryCount: t,
                                                    d: D,
                                                    intermediate: cfg.intermediateSize,
                                                    xStrideElements: D,
                                                    yStrideElements: D)
                for row in 0..<t {
                    int8.encode(commandBuffer: sharedCB,
                                weights: v.sharedGate.buffer,
                                weightsOffset: Int(v.sharedGate.offset),
                                scales: v.sharedGate.buffer,
                                scalesOffset: Int(v.sharedGate.scaleOffset),
                                biases: v.sharedGate.buffer,
                                biasesOffset: Int(v.sharedGate.biasOffset),
                                x: scratch.qwenNormed2,
                                xOffset: row * D * MemoryLayout<Float16>.stride,
                                y: scratch.qwenGateLogits,
                                yOffset: row * MemoryLayout<Float16>.stride,
                                m: 1, n: UInt32(D))
                }
                if L == 0, let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                    Self.debugDumpBuffer(scratch.qwenGateLogits, elements: t,
                                         to: "\(dumpDir)/l0_gatelogs.f16", commandBuffer: sharedCB)
                }
                laKernels.encodeSharedGateMul(
                    commandBuffer: sharedCB,
                    sharedOut: scratch.h1,
                    gateLogit: scratch.qwenGateLogits,
                    t: UInt32(t), d: UInt32(D))
                sharedCB.commit()
                try waitForCompletion(sharedCB)

                let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                    device: ctx.device,
                    routes: routes)
                let routedOffsets = model.routedExpertOffsets(layer: L)
                struct PendingPrefillTile {
                    let tileIndex: Int
                    let commandBuffer: MTLCommandBuffer
                    let fetch: PrefillStreamedTileFetchResult
                    let argumentBuffer: PrefillStreamedTileArgumentBuffer
                }
                var pendingTiles: [PendingPrefillTile] = []
                var tileLifetime = PrefillStreamedTileSlotLifetime()
                func drainOldestPendingTile() throws {
                    guard !pendingTiles.isEmpty else { return }
                    let pending = pendingTiles.removeFirst()
                    try withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                        try waitForCompletion(pending.commandBuffer)
                    }
                    if !pending.fetch.plannedMissSlots.isEmpty {
                        try tileLifetime.complete(tileIndex: pending.tileIndex)
                    }
                }

                let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                for (tileIndex, tile) in routes.tiles.enumerated() {
                    let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                        forTile: tileIndex,
                        routes: routes)
                    var plannedFetch: RoutedExpertFetchPlan?
                    if !pendingTiles.isEmpty {
                        let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                        if !pendingAssignedSlots.isEmpty {
                            let pendingSlots = Set(pendingAssignedSlots)
                            let plan = try model.planRoutedExpertsIfPossible(
                                layer: L,
                                experts: expertIDs,
                                avoidingSlots: pendingSlots)
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: true,
                                    pendingDepth: pendingTiles.count,
                                    pendingAssignedSlots: pendingAssignedSlots,
                                    avoidingSlotPlanAvailable: plan != nil))
                            switch decision {
                            case .prefetchNext:
                                guard let plan else {
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler requested missing plan")
                                }
                                plannedFetch = plan
                            case .drainBeforeIssue:
                                try drainOldestPendingTile()
                            case .issueWithoutPending:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler ignored pending tile")
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: true,
                                    pendingDepth: pendingTiles.count,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .drainBeforeIssue:
                                try drainOldestPendingTile()
                            case .issueWithoutPending, .prefetchNext:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler failed to drain empty-slot pending tile")
                            }
                        }
                    } else {
                        let decision = routedTileScheduler.decide(
                            PrefillRoutedTileSchedulerInput(
                                hasPendingTile: false,
                                pendingAssignedSlots: [],
                                avoidingSlotPlanAvailable: false))
                        switch decision {
                        case .issueWithoutPending:
                            break
                        case .prefetchNext, .drainBeforeIssue:
                            throw ModelError.indexCorrupt(
                                detail: "routed tile scheduler requested pending action without pending tile")
                        }
                    }
                    let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                        model: model,
                        layer: L,
                        tileIndex: tileIndex,
                        routes: routes,
                        plannedFetch: plannedFetch,
                        avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                    try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                          pairStart: Int(tile.pairStart),
                                                          pairCount: Int(tile.pairCount))
                    if !fetch.plannedMissSlots.isEmpty {
                        try tileLifetime.begin(tileIndex: tileIndex,
                                               plannedSlots: fetch.plannedMissSlots)
                    }
                    let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                        device: ctx.device,
                        binding: fetch.binding)
                    let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                        pairStart: tile.pairStart,
                        pairCount: tile.pairCount,
                        d: UInt32(D),
                        routedIntermediate: UInt32(cfg.moeIntermediateSize),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D),
                        binding: fetch.binding,
                        offsets: routedOffsets)
                    guard let tileCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    tileCB.label =
                        "prefill start=\(startPosition) count=\(t) layer=\(L) phase=routed_tile"
                    _ = prefillGroupedMoE.encodeStreamedBatched(
                        commandBuffer: tileCB,
                        hidden: scratch.qwenNormed2,
                        sortedPairs: metadata.sortedPairs,
                        routePartials: scratch.routePartials,
                        gateUpActScratch: scratch.routedGateUpActScratch,
                        downScratch: scratch.routedDownScratch,
                        argumentBuffer: argumentBuffer,
                        binding: fetch.binding,
                        params: streamedParams,
                        pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                    tileCB.commit()
                    pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                           commandBuffer: tileCB,
                                                           fetch: fetch,
                                                           argumentBuffer: argumentBuffer))
                    while pendingTiles.count > schedulerConfig.maxPendingDepth {
                        try drainOldestPendingTile()
                    }
                }
                while !pendingTiles.isEmpty {
                    try drainOldestPendingTile()
                }
                guard let tailCB = ctx.queue.makeCommandBuffer() else {
                    throw ModelError.residentBufferWrapFailed
                }
                prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                  routePartials: scratch.routePartials,
                                                  routeWeights: scratch.routeWeights,
                                                  h2: scratch.h2,
                                                  queryCount: UInt32(t),
                                                  topK: UInt32(cfg.topKExperts),
                                                  d: UInt32(D))
                if (L == 0 || L == 3), let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                    Self.debugDumpBuffer(scratch.h1, elements: t * D,
                                         to: "\(dumpDir)/l\(L)_h1.f16", commandBuffer: tailCB)
                    Self.debugDumpBuffer(scratch.h2, elements: t * D,
                                         to: "\(dumpDir)/l\(L)_h2.f16", commandBuffer: tailCB)
                }
                laKernels.encodeLayerTail(commandBuffer: tailCB,
                                          hidden: scratch.hidden, hiddenOffset: 0,
                                          h1: scratch.h1, h1Offset: 0,
                                          h2: scratch.h2, h2Offset: 0,
                                          t: UInt32(t), d: UInt32(D))
                if let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                    let tailDumpCB = tailCB
                    Self.debugDumpBuffer(scratch.hidden,
                                         elements: t * D,
                                         to: "\(dumpDir)/layer_\(L).f16",
                                         commandBuffer: tailDumpCB)
                    if (L == 0 || L == 2), let stateBuf = try kv?.stateBuffer(layer: L) {
                        let dumpCB = ctx.queue.makeCommandBuffer()!
                        Self.debugDumpFP32(stateBuf,
                                           to: "\(dumpDir)/l\(L)_state_after.f32",
                                           commandBuffer: dumpCB)
                        dumpCB.commit()
                        try await waitForCompletion(dumpCB)
                    }
                }
                tailCB.commit()
                try withExtendedLifetime(metadata) {
                    try waitForCompletion(tailCB)
                }
                if L + 1 < cfg.numLayers {
                    guard let nextCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    nextCB.label =
                        "prefill start=\(startPosition) count=\(t) layer=\(L + 1) phase=qkv_attention"
                    cb = nextCB
                }
                continue
            }
            let views = layerViews[L]
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.inputNorm.buffer,
                                   weightOffset: Int(views.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .q,
                                 weights: views.q,
                                 x: scratch.normed,
                                 y: scratch.q,
                                 rows: qDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: qDim)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: views.k,
                                 x: scratch.normed,
                                 y: scratch.kStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: views.v,
                                 x: scratch.normed,
                                 y: scratch.vStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)

            let rotatedPairs = isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDim / 2)
            prefillQKVEpilogue.encode(commandBuffer: cb,
                                       q: scratch.q,
                                       k: scratch.kStage,
                                       v: scratch.vStage,
                                       qWeight: views.qNorm.buffer,
                                       qWeightOffset: Int(views.qNorm.offset),
                                       kWeight: views.kNorm.buffer,
                                       kWeightOffset: Int(views.kNorm.offset),
                                       startPosition: UInt32(startPosition),
                                       queryCount: UInt32(t),
                                       headDim: UInt32(headDim),
                                       numQHeads: UInt32(cfg.numHeads),
                                       numKVHeads: UInt32(numKVHeads),
                                       qTokenStrideElements: UInt32(qDim),
                                       kvTokenStrideElements: UInt32(kvDim),
                                       theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                       rotatedPairs: rotatedPairs,
                                       eps: eps)

            if let kv {
                let bytes = t * kvDim * MemoryLayout<Float16>.stride
                try copyPrefillKVToCache(commandBuffer: cb,
                                         kv: kv,
                                         layer: L,
                                         startPosition: startPosition,
                                         tokenCount: t,
                                         keySource: scratch.kStage,
                                         valueSource: scratch.vStage,
                                         bytesPerToken: bytes / t)
            }
            let params = PrefillAttentionParams(
                    startPosition: UInt32(startPosition),
                    queryCount: UInt32(t),
                    headDim: UInt32(headDim),
                    numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(numKVHeads),
                    kvValidCount: UInt32(startPosition + t),
                    slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                    kvTokenStrideElements: UInt32(kvDim),
                    qTokenStrideElements: UInt32(qDim),
                    oTokenStrideElements: UInt32(qDim),
                    scale: 1.0,
                    bidirectionalBlockStart: UInt32(
                        isFull ? 0 : bidirectionalBlock?.lowerBound ?? 0),
                    bidirectionalBlockEnd: UInt32(
                        isFull ? 0 : bidirectionalBlock?.upperBound ?? 0))
            if let kv {
                    let keyBuffer = kv.keyBuffer(layer: L, validTokenCount: startPosition + t)
                    let valueBuffer = kv.valueBuffer(layer: L, validTokenCount: startPosition + t)
                    let ringCapacity = kv.ringCapacity(layer: L)
                    let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    prefillAttention.encodeCausal(commandBuffer: cb,
                                                  q: scratch.q,
                                                  k: keyBuffer,
                                                  v: valueBuffer,
                                                  out: scratch.attentionOutput,
                                                  params: params,
                                                  kvRingCapacity: activeRingCapacity,
                                                  layerKind: isFull ? .full : .slidingWindow,
                                                  path: prefillAttentionPath)
            } else {
                throw PrefillError.chunkedUnsupported(
                    "chunked prefill attention requires FP16 KV")
            }
            encodeInt4Projection(commandBuffer: cb,
                                     family: .o,
                                     weights: views.o,
                                     x: scratch.attentionOutput,
                                     y: scratch.h1,
                                     rows: D,
                                     columns: qDim,
                                     tokenCount: t,
                                     xStrideElements: qDim,
                                     yStrideElements: D)
            prefillPostAttention.encode(commandBuffer: cb,
                                            hidden: scratch.hidden,
                                            attn: scratch.h1,
                                            denseX: scratch.denseX,
                                            routedX: scratch.routedX,
                                            routerX: scratch.routerX,
                                            postAttentionWeight: views.postAttention.buffer,
                                            postAttentionWeightOffset: Int(views.postAttention.offset),
                                            preFFNWeight: views.preFFN.buffer,
                                            preFFNWeightOffset: Int(views.preFFN.offset),
                                            preFFN2Weight: views.preFFN2.buffer,
                                            preFFN2WeightOffset: Int(views.preFFN2.offset),
                                            queryCount: UInt32(t),
                                            d: UInt32(D),
                                            hiddenStrideElements: UInt32(D),
                                            attnStrideElements: UInt32(D),
                                            denseStrideElements: UInt32(D),
                                            routedStrideElements: UInt32(D),
                                            routerStrideElements: UInt32(D),
                                            eps: eps)
            prefillRouter.encodeGemma4Block(
                        commandBuffer: cb,
                        weights: views.router.buffer,
                        weightsOffset: Int(views.router.offset),
                        scales: views.router.buffer,
                        scalesOffset: Int(views.router.scaleOffset),
                        biases: views.router.buffer,
                        biasesOffset: Int(views.router.biasOffset),
                        hidden: scratch.routerX,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: views.routerPerExpertScale.buffer,
                        perExpertScaleOffset: Int(views.routerPerExpertScale.offset),
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))

                    cb.commit()
                    try waitForCompletion(cb)

                    let routeCount = t * cfg.topKExperts
                    let idPtr = scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    var routeIDs = [UInt32]()
                    routeIDs.reserveCapacity(routeCount)
                    var routeWeights = [Float16]()
                    routeWeights.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        routeWeights.append(weightPtr[i])
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                                   weights: routeWeights,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                    let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                    let routeTileExpertCount: Int
                    if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
                        guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                            throw PrefillError.chunkedUnsupported(
                                "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
                        }
                        routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
                    } else {
                        routeTileExpertCount = schedulerConfig.tileExperts
                    }
                    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: routeTileExpertCount,
                        expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

                    guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    sharedCB.label =
                        "prefill start=\(startPosition) count=\(t) layer=\(L) phase=shared_expert"
                    let sharedProj = sharedExpertProjections[L]
                    try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                        x: scratch.denseX,
                                                        y: scratch.h1,
                                                        gate: sharedProj.gate,
                                                        up: sharedProj.up,
                                                        down: sharedProj.down,
                                                        scratchGate: scratch.sharedGateScratch,
                                                        scratchUp: scratch.sharedUpScratch,
                                                        scratchAct: scratch.sharedActScratch,
                                                        queryCount: t,
                                                        d: D,
                                                        intermediate: cfg.intermediateSize,
                                                        xStrideElements: D,
                                                        yStrideElements: D)
                    prefillRMS.encodeBF16W(commandBuffer: sharedCB,
                                           x: scratch.h1,
                                           weight: sharedProj.postF1!.buffer,
                                           weightOffset: Int(sharedProj.postF1!.offset),
                                           out: scratch.h1,
                                           t: UInt32(t),
                                           d: UInt32(D),
                                           eps: eps)
                    sharedCB.commit()
                    try waitForCompletion(sharedCB)

                    let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    let routedOffsets = model.routedExpertOffsets(layer: L)
                    struct PendingPrefillTile {
                        let tileIndex: Int
                        let commandBuffer: MTLCommandBuffer
                        let fetch: PrefillStreamedTileFetchResult
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                    }
                    var pendingTiles: [PendingPrefillTile] = []
                    var tileLifetime = PrefillStreamedTileSlotLifetime()
                    func drainOldestPendingTile() throws {
                        guard !pendingTiles.isEmpty else { return }
                        let pending = pendingTiles.removeFirst()
                        try withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                            try waitForCompletion(pending.commandBuffer)
                        }
                        if !pending.fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                    for (tileIndex, tile) in routes.tiles.enumerated() {
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                        var plannedFetch: RoutedExpertFetchPlan?
                        if !pendingTiles.isEmpty {
                            let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                            if !pendingAssignedSlots.isEmpty {
                                let pendingSlots = Set(pendingAssignedSlots)
                                let plan = try model.planRoutedExpertsIfPossible(
                                    layer: L,
                                    experts: expertIDs,
                                    avoidingSlots: pendingSlots)
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: pendingAssignedSlots,
                                        avoidingSlotPlanAvailable: plan != nil))
                                switch decision {
                                case .prefetchNext:
                                    guard let plan else {
                                        throw ModelError.indexCorrupt(
                                            detail: "routed tile scheduler requested missing plan")
                                    }
                                    plannedFetch = plan
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler ignored pending tile")
                                }
                            } else {
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: [],
                                        avoidingSlotPlanAvailable: false))
                                switch decision {
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending, .prefetchNext:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler failed to drain empty-slot pending tile")
                                }
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: false,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .issueWithoutPending:
                                break
                            case .prefetchNext, .drainBeforeIssue:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler requested pending action without pending tile")
                            }
                        }
                        let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                            model: model,
                            layer: L,
                            tileIndex: tileIndex,
                            routes: routes,
                            plannedFetch: plannedFetch,
                            avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        if !fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedMissSlots)
                        }
                        let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                            device: ctx.device,
                            binding: fetch.binding)
                        let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: fetch.binding,
                            offsets: routedOffsets)
                        guard let tileCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        tileCB.label =
                            "prefill start=\(startPosition) count=\(t) layer=\(L) phase=routed_tile"
                        _ = prefillGroupedMoE.encodeStreamedBatched(
                            commandBuffer: tileCB,
                            hidden: scratch.routedX,
                            sortedPairs: metadata.sortedPairs,
                            routePartials: scratch.routePartials,
                            gateUpActScratch: scratch.routedGateUpActScratch,
                            downScratch: scratch.routedDownScratch,
                            argumentBuffer: argumentBuffer,
                            binding: fetch.binding,
                            params: streamedParams,
                            pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    guard let tailCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                      routePartials: scratch.routePartials,
                                                      routeWeights: scratch.routeWeights,
                                                      h2: scratch.h2,
                                                      queryCount: UInt32(t),
                                                      topK: UInt32(cfg.topKExperts),
                                                      d: UInt32(D))
                    let scalarBits = views.layerScalar.buffer.contents()
                        .advanced(by: Int(views.layerScalar.offset))
                        .assumingMemoryBound(to: UInt16.self)[0]
                    prefillLayerTail.encode(commandBuffer: tailCB,
                                            h2: scratch.h2,
                                            h1: scratch.h1,
                                            hidden: scratch.hidden,
                                            postFFN2Weight: views.postFFN2.buffer,
                                            postFFN2WeightOffset: Int(views.postFFN2.offset),
                                            postFFNWeight: views.postFFN.buffer,
                                            postFFNWeightOffset: Int(views.postFFN.offset),
                                            queryCount: UInt32(t),
                                            d: UInt32(D),
                                            h2StrideElements: UInt32(D),
                                            h1StrideElements: UInt32(D),
                                            hiddenStrideElements: UInt32(D),
                                            eps: eps,
                                            layerScalar: Quantization.bf16ToFloat(scalarBits))
                    tailCB.commit()
                    try withExtendedLifetime(metadata) {
                        try waitForCompletion(tailCB)
                    }
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        nextCB.label =
                            "prefill start=\(startPosition) count=\(t) layer=\(L + 1) phase=qkv_attention"
                        cb = nextCB
                    }
                    continue
        }

        if writeFinalHead {
            let finalNorm = model.finalNorm
            let lm = model.lmHead
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            finalCB.label = "prefill start=\(startPosition) count=\(t) phase=final_head"
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                fusionHead.encodeGreedyDecode(
                    commandBuffer: finalCB,
                    hidden: scratch.hidden,
                    hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lm.buffer,
                    weightsOffset: Int(lm.offset),
                    scales: lm.buffer,
                    scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer,
                    biasesOffset: Int(lm.biasOffset),
                    outToken: greedyTokenBuf,
                    d: UInt32(D),
                    vocab: UInt32(cfg.vocabSize),
                    rmsEps: eps)
            } else {
                prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                                 hiddenBlock: scratch.hidden,
                                                 row: t - 1,
                                                 rowStrideElements: D,
                                                 normWeight: finalNorm.buffer,
                                                 normWeightOffset: Int(finalNorm.offset),
                                                 weights: lm.buffer,
                                                 weightsOffset: Int(lm.offset),
                                                 scales: lm.buffer,
                                                 scalesOffset: Int(lm.scaleOffset),
                                                 biases: lm.buffer,
                                                 biasesOffset: Int(lm.biasOffset),
                                                 logits: logits,
                                                 d: UInt32(D),
                                                 vocab: UInt32(cfg.vocabSize),
                                                 rmsEps: eps)
            }
            finalCB.commit()
            try waitForCompletion(finalCB)
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }

        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    static func debugDumpFP32(_ buffer: MTLBuffer, to path: String,
                              commandBuffer: MTLCommandBuffer) {
        guard let stage = buffer.device.makeBuffer(length: buffer.length,
                                                   options: .storageModeShared),
              let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: buffer, sourceOffset: 0,
                  to: stage, destinationOffset: 0, size: buffer.length)
        blit.endEncoding()
        // Retain the staging buffer until the handler runs: a second dump on
        // the same command buffer would otherwise reuse (and clobber) it.
        commandBuffer.addCompletedHandler { _ in
            let data = Data(bytes: stage.contents(), count: buffer.length)
            try? data.write(to: URL(fileURLWithPath: path))
            _ = stage
        }
    }

    static func debugDumpBuffer(_ buffer: MTLBuffer, elements: Int, to path: String,
                                commandBuffer: MTLCommandBuffer) {
        let bytes = elements * MemoryLayout<Float16>.stride
        guard bytes <= buffer.length,
              let stage = buffer.device.makeBuffer(length: bytes, options: .storageModeShared),
              let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: buffer, sourceOffset: 0,
                  to: stage, destinationOffset: 0, size: bytes)
        blit.endEncoding()
        commandBuffer.addCompletedHandler { _ in
            let data = Data(bytes: stage.contents(), count: bytes)
            try? data.write(to: URL(fileURLWithPath: path))
        }
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
                let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
        let sqrtHidden = cfg.embedOutScale
        if cfg.usesLinearAttention {
            try await produceTokenQwen(token: token, position: position,
                                       into: logits, emitHead: emitHead,
                                       outputMode: outputMode)
            return
        }
        struct PendingRoutedCommand {
            let cb: MTLCommandBuffer
            let sharedCB: MTLCommandBuffer?
            let phase1HitCB: MTLCommandBuffer?
            let encodeAndCommitNanos: UInt64
        }
        var pendingRoutedCommand: PendingRoutedCommand?

        func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                        waitIfNeeded: Bool) throws {
            if waitIfNeeded {
                if let sharedCB = pending.sharedCB {
                    waitUntilCompleted(sharedCB)
                }
                if let phase1HitCB = pending.phase1HitCB {
                    waitUntilCompleted(phase1HitCB)
                }
                waitUntilCompleted(pending.cb)
            }
            if let sharedCB = pending.sharedCB {
                try checkCommandBufferError(sharedCB)
            }
            if let phase1HitCB = pending.phase1HitCB {
                try checkCommandBufferError(phase1HitCB)
            }
            try checkCommandBufferError(pending.cb)
            totalCb2Nanos &+= pending.encodeAndCommitNanos
        }

        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }

        // Embed lookup + sqrt(H) fused.
        let emb = model.embedding
        do {
            try runSync { cb in
                embedInt4.encode(commandBuffer: cb,
                                 table:  emb.buffer, tableOffset:  Int(emb.offset),
                                 scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                 biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                 out: hidden,
                                 tokenId: UInt32(bitPattern: token),
                                 d: D,
                                 outScale: sqrtHidden)
            }
        }

        for L in 0..<cfg.numLayers {
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim     = UInt32(cfg.numHeads * headDimL)
            let kvDim    = UInt32(numKVL * headDimL)
            let kSlot    = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
            let vSlot    = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
            let seqLen   = UInt32(position + 1)

            let inNorm   = try model.inputNorm(layer: L)
            let q        = try model.qProj(layer: L)
            let k        = try model.kProj(layer: L)
            // v_proj only exists on SWA layers; full layers reuse k_proj.
            let vProj    = isFull ? k : (try model.vProj(layer: L))
            let o        = try model.oProj(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let qNorm    = try model.qNorm(layer: L)
            let kNorm    = try model.kNorm(layer: L)
            let preFFN   = try model.preFFN(layer: L)
            let preFFN2  = try model.preFFN2(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let postF2   = try model.postFFN2(layer: L)
            let postF    = try model.postFFN(layer: L)
            let routerW  = try model.router(layer: L)
            let perExpertScale = try model.routerPerExpertScale(layer: L)
            let layerScalarView = try model.layerScalar(layer: L)

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Everything up to and including the router runs in a single CB:
            // the only reason to break is the CPU readback of router indices
            // needed to issue I/O for the routed-expert blobs.
            let gInputNorm: (MTLCommandBuffer) -> Void = { [self] cb in
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                out: normed,
                                d: D, eps: eps)
            }

            let gQKV: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedQKVGEMV.encode(commandBuffer: cb,
                                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                    x: normed,
                                    qOut: qScratch,
                                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                    qRows: qDim,
                                    kvRows: kvDim,
                                    n: D)
            }

            let gQKVEpilogue: (MTLCommandBuffer) -> Void = { [self] cb in
                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                fusedQKVEpilogue.encode(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer,
                                        kOffset: kSlot.offset,
                                        v: vSlot.buffer,
                                        vOffset: vSlot.offset,
                                        qWeight: qNorm.buffer,
                                        qWeightOffset: Int(qNorm.offset),
                                        kWeight: kNorm.buffer,
                                        kWeightOffset: Int(kNorm.offset),
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        position: UInt32(position),
                                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                        rotatedPairs: rotated,
                                        eps: eps)
            }

            let gAttention: (MTLCommandBuffer) -> Void = { [self] cb in
                guard kv != nil else {
                    preconditionFailure("FP16 attention requires an FP16 KV cache")
                }
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: 1.0)
                } else {
                    let ringCapacity = kv?.ringCapacity(layer: L) ?? 0
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer, kOffset: 0,
                                        v: vSlot.buffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: 1.0,
                                        ringCapacity: activeRingCapacity)
                }
            }
            let gOProj: (MTLCommandBuffer) -> Void = { [self] cb in
                int4.encode(commandBuffer: cb,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                            biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim)
            }

            let gPostAttnSetup: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedPostAttentionSetup.encode(commandBuffer: cb,
                                               hidden: hidden,
                                               attn: oOut,
                                               denseX: denseX,
                                               routedX: routedX,
                                               routerX: routerInput,
                                               postAttentionWeight: postAttn.buffer,
                                               postAttentionWeightOffset: Int(postAttn.offset),
                                               preFFNWeight: preFFN.buffer,
                                               preFFNWeightOffset: Int(preFFN.offset),
                                               preFFN2Weight: preFFN2.buffer,
                                               preFFN2WeightOffset: Int(preFFN2.offset),
                                               d: D,
                                               eps: eps)
            }

            let gRouter: (MTLCommandBuffer) -> Void = { [self] cb in
                moe.encodeRouterGemma4(commandBuffer: cb,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                    biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                    hidden: routerInput,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: perExpertScale.buffer,
                    perExpertScaleOffset: Int(perExpertScale.offset),
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
            }

            let cb = ctx.queue.makeCommandBuffer()!
            gInputNorm(cb)
            gQKV(cb)
            gQKVEpilogue(cb)
            gAttention(cb)
            gOProj(cb)
            gPostAttnSetup(cb)
            gRouter(cb)
            cb.commit()
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            waitUntilCompleted(cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            if let pending = pendingRoutedCommand {
                try finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            try checkCommandBufferError(cb)
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            // CPU readback to fetch routed-expert blobs from disk.
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }

            let routedOffsets = model.routedExpertOffsets(layer: L)
            let topK = UInt32(cfg.topKExperts)
            let canPlanPhase1HitSplit =
                cfg.topKExperts <= MoE.maxStreamedExperts
            let plannedFetch = canPlanPhase1HitSplit
                ? try model.planRoutedExperts(layer: L, experts: experts)
                : nil
            var phase1HitCB: MTLCommandBuffer?
            var phase1HitSplitArgBuf: MTLBuffer?
            var phase1HitSplitRoutedBufs: [MTLBuffer] = []
            var phase1HitSlots: [UInt32] = []
            var phase1MissSlots: [UInt32] = []

            if let plan = plannedFetch {
                let missSet = Set(plan.misses)
                phase1HitSlots = (0..<cfg.topKExperts)
                    .filter { !missSet.contains($0) }
                    .map { UInt32($0) }
                phase1MissSlots = plan.misses.map { UInt32($0) }
            }
            func encodeRoutedPhase1Full(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer]
            ) {
                moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                        routedArgBuffer: argBuf,
                                                        routedBlobs: routedBufs,
                                                        routedOffsets: routedOffsets,
                                                        x: routedX,
                                                        acts: moeActs,
                                                        d: D,
                                                        f: FmoE,
                                                        topK: topK)
            }

            func encodeRoutedPhase1Subset(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer],
                activeSlots: MTLBuffer,
                activeSlotIndices: [UInt32],
                activeCount: UInt32
            ) {
                moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: cb,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX,
                    acts: moeActs,
                    activeSlots: activeSlots,
                    activeSlotIndices: activeSlotIndices,
                    activeCount: activeCount,
                    d: D,
                    f: FmoE,
                    topK: topK)
            }

            if let plan = plannedFetch,
               plan.hits > 0,
               !plan.misses.isEmpty {
                let plannedBlobs = try model.routedExpertBuffers(for: plan)
                phase1HitSplitRoutedBufs = plannedBlobs.map { $0.buffer }
                phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                    routedBlobs: phase1HitSplitRoutedBufs,
                    topK: topK)
                if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                    writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                    let cb = ctx.queue.makeCommandBuffer()!
                    encodeRoutedPhase1Subset(
                        cb,
                        argBuf: argBuf,
                        routedBufs: phase1HitSplitRoutedBufs,
                        activeSlots: moeHitActiveSlots,
                        activeSlotIndices: phase1HitSlots,
                        activeCount: UInt32(phase1HitSlots.count))
                    phase1HitCB = cb
                }
            }

            // The shared dense MLP depends only on denseX, not on the routed
            // experts. Commit it without waiting so its GPU work overlaps the
            // routed-expert pread. The routed CB follows it on the same queue,
            // so the combine sees h1Buf.
            let gSharedFFN: (MTLCommandBuffer) -> Void = { [self] cb in
                try! shared.encode(commandBuffer: cb,
                                   x: denseX,
                                   gate: sharedProj.gate,
                                   up: sharedProj.up,
                                   down: sharedProj.down,
                                   y: h1Buf,
                                   scratchGate: denseScratchGate,
                                   scratchUp: denseScratchUp,
                                   scratchAct: denseScratchAct)
            }
            let gSharedNorm: (MTLCommandBuffer) -> Void = { [self] cb in
                rms.encodeBF16W(commandBuffer: cb, x: h1Buf,
                                weight: sharedProj.postF1!.buffer,
                                weightOffset: Int(sharedProj.postF1!.offset),
                                out: h1Buf, d: D, eps: eps)
            }
            let sharedCB = ctx.queue.makeCommandBuffer()!
            gSharedFFN(sharedCB)
            gSharedNorm(sharedCB)
            sharedCB.commit()
            if let cb = phase1HitCB {
                cb.commit()
            }
            if rdadviseEnabled && rdadvisePolicyMode != .off {
                let requestedMisses = plannedFetch?.misses.count ?? experts.count
                let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                    layer: L,
                    missCount: requestedMisses)
                if let skipped = shouldSkipRDAdvice(position: position,
                                                    requestedMisses: requestedMisses,
                                                    estimatedBytes: estimatedAdviceBytes,
                                                    canOverlapUsefulGPUWork: true) {
                    recordRDAdvice(skipped, wallNanos: 0)
                } else {
                    let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let result: ExpertIOAdviceResult
                    if let plannedFetch {
                        result = try model.adviseRoutedExperts(plan: plannedFetch)
                    } else {
                        result = try model.adviseRoutedExperts(layer: L, experts: experts)
                    }
                    let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                    recordRDAdvice(result, wallNanos: wallNanos)
                    updateRDAdvicePolicy(after: result, position: position)
                }
            }

            // Routed-expert pread — overlaps the shared MLP GPU work above.
            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs: [TensorView]
            if let plannedFetch {
                blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
            } else {
                blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
            }
            let layerIo = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
            totalIoNanos &+= layerIo
            let routedBufs = blobs.map { $0.buffer }
            let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let scalarPtr = layerScalarView.buffer.contents()
                .advanced(by: Int(layerScalarView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let layerScalar = Quantization.bf16ToFloat(scalarPtr[0])

            let gTail: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedTail.encode(commandBuffer: cb,
                                 h2: h2Buf,
                                 h1: h1Buf,
                                 hidden: hidden,
                                 postFFN2Weight: postF2.buffer,
                                 postFFN2WeightOffset: Int(postF2.offset),
                                 postFFNWeight: postF.buffer,
                                 postFFNWeightOffset: Int(postF.offset),
                                 d: D,
                                 eps: eps,
                                 layerScalar: layerScalar)
            }
            let routedCB = ctx.queue.makeCommandBuffer()!
            let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
                ? phase1HitSplitArgBuf
                : nil
            let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs,
                topK: topK)
            if splitArgBuf != nil {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                encodeRoutedPhase1Subset(
                    routedCB,
                    argBuf: argBuf,
                    routedBufs: routedBufs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count))
            } else {
                encodeRoutedPhase1Full(routedCB,
                                       argBuf: argBuf,
                                       routedBufs: routedBufs)
            }
            moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: routedBufs,
                                                   routedOffsets: routedOffsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: zeroResidual,
                                                   y: h2Buf,
                                                   d: D,
                                                   f: FmoE,
                                                   topK: topK)
            gTail(routedCB)
            routedCB.commit()
            precondition(pendingRoutedCommand == nil,
                         "routed command-buffer pipeline drained before queuing the next layer")
            pendingRoutedCommand = PendingRoutedCommand(
                cb: routedCB,
                sharedCB: sharedCB,
                phase1HitCB: phase1HitCB,
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
            continue
        }
        if let pending = pendingRoutedCommand {
            try finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = model.finalNorm
        let lm    = model.embedding
        let gFinalNorm: (MTLCommandBuffer) -> Void = { cb in
            self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                 weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                 out: self.normed, d: D, eps: eps)
        }
        let gLmHead: (MTLCommandBuffer) -> Void = { cb in
            self.int4.encode(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) -> Void = { cb in
            self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if useFusedHeadForThisToken {
                try runSync(gFusionHead)
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
                if let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                    let line = "greedy:\\(lastGreedyToken)\\n"
                    let fd = open(dumpDir + "/greedy.txt", O_WRONLY | O_CREAT | O_APPEND, 0o644)
                    if fd >= 0 { _ = write(fd, line, line.utf8.count); close(fd) }
                }
            } else {
                try runSync { cb in
                    gFinalNorm(cb)
                    gLmHead(cb)
                }
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        kv?.advance()
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        body(cb)
        cb.commit()
        try waitForCompletion(cb)
    }

    // MARK: - Qwen3.6 hybrid decode
    //
    //   embed_lookup_int4(token)            (no sqrt(H) scale)
    //   for L in 0..<40:
    //     a = rmsnorm_bf16w(h, input_layernorm)
    //     full layers:
    //       qkv GEMV -> [16 heads x (q 256 | gate 256)], k, v (real V)
    //       qwen_attn_epilogue: per-head q/k RMSNorm + partial RoPE + sigmoid gate
    //       attention(scale = head_dim^-0.5, causal)
    //     linear layers (gated delta rule, fp32 state):
    //       qkv = in_proj_qkv(a); z = in_proj_z(a); a/b = in_proj_{a,b}(a)
    //       conv1d + silu -> l2norm q/k -> GDN recurrence -> silu(z)*rmsnorm(y)
    //       out_proj
    //     h += attn_or_gdn_out
    //     normed2 = rmsnorm_bf16w(h, post_attention_layernorm)
    //     router  = top-8 of softmax(mlp.gate GEMV)   (no per-expert scale)
    //     shared  = sigmoid(shared_expert_gate GEMV) * SharedExpertFFN(normed2)
    //     routed  = Σ top-8 softmax weight * ExpertFFN(normed2)
    //     h += shared + routed
    //   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), lm_head^T)
    //
    // Command-buffer structure mirrors the Gemma path: CB1 through the router
    // (CPU readback of indices), shared-expert CB overlapped with the routed
    // blob pread, then the routed CB whose tail feeds the next layer's CB1.

    private func produceTokenQwen(token: Int32,
                                  position: Int,
                                  into logits: MTLBuffer,
                                  emitHead: Bool,
                                  outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        let D = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
        let nk = UInt32(cfg.linearNumKeyHeads)
        let nv = UInt32(cfg.linearNumValueHeads)
        let dk = UInt32(cfg.linearKeyHeadDim)
        let dv = UInt32(cfg.linearValueHeadDim)
        let convDim = nk * dk * 2 + nv * dv
        let qRows = UInt32(cfg.numHeads * cfg.headDim * 2)   // q + gate
        let kvRows = UInt32(cfg.numFullKVHeads * cfg.fullHeadDim)
        let seqLen = UInt32(position + 1)
        let rotatedPairs = UInt32(Double(cfg.headDim) * cfg.partialRotaryFactor / 2.0)
        let attnScale = cfg.attentionScale

        struct Pending {
            let cb: MTLCommandBuffer
            let sharedCB: MTLCommandBuffer
        }
        var pending: Pending?

        func drainPending(wait: Bool) throws {
            if let p = pending {
                if wait { waitUntilCompleted(p.sharedCB) }
                try checkCommandBufferError(p.sharedCB)
                if wait { waitUntilCompleted(p.cb) }
                try checkCommandBufferError(p.cb)
                pending = nil
            }
        }

        // Embed lookup (no Gemma sqrt(H) scale).
        let emb = model.embedding
        try runSync { cb in
            self.embedInt4.encode(commandBuffer: cb,
                                  table: emb.buffer, tableOffset: Int(emb.offset),
                                  scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                  biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                  out: self.hidden,
                                  tokenId: UInt32(bitPattern: token),
                                  d: D,
                                  outScale: cfg.embedOutScale)
        }

        for L in 0..<cfg.numLayers {
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let inNorm = try model.inputNorm(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let routerW = try model.router(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let sharedGateView = try model.sharedExpertScalarGate(layer: L)

            let cb = ctx.queue.makeCommandBuffer()!
            cb.label = "qwen decode L\(L) pos=\(position) cb1"
            if L == 0, let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"],
               let cw = try kv?.convWindowBuffer(layer: 0) {
                let wdata = Data(bytes: cw.contents(), count: cw.length)
                try? wdata.write(to: URL(fileURLWithPath: "\(dumpDir)/d0_window_before.f16"))
                if let sb = try kv?.stateBuffer(layer: 0) {
                    let sdata = Data(bytes: sb.contents(), count: sb.length)
                    try? sdata.write(to: URL(fileURLWithPath: "\(dumpDir)/d0_state_before.f32"))
                }
            }
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed,
                            d: D, eps: eps)
            if isFull {
                let q = try model.qProj(layer: L)
                let k = try model.kProj(layer: L)
                let v = try model.vProj(layer: L)
                let o = try model.oProj(layer: L)
                let qNorm = try model.qNorm(layer: L)
                let kNorm = try model.kNorm(layer: L)
                let kSlot = kv?.kSlot(layer: L, position: position)
                    ?? (buffer: kStage, offset: 0)
                let vSlot = kv?.vSlot(layer: L, position: position)
                    ?? (buffer: vStage, offset: 0)

                func gemv(_ view: TensorView, x: MTLBuffer, y: MTLBuffer,
                          yOffset: Int = 0, m: UInt32, n: UInt32) {
                    int4.encode(commandBuffer: cb,
                                weights: view.buffer, weightsOffset: Int(view.offset),
                                scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                                biases: view.buffer, biasesOffset: Int(view.biasOffset),
                                x: x, y: y, yOffset: yOffset,
                                m: m, n: n)
                }
                gemv(q, x: normed, y: qkvBuf, m: qRows, n: D)
                gemv(k, x: normed, y: kSlot.buffer, yOffset: kSlot.offset, m: kvRows, n: D)
                gemv(v, x: normed, y: vSlot.buffer, yOffset: vSlot.offset, m: kvRows, n: D)
                laKernels.encodeAttnEpilogue(
                    commandBuffer: cb,
                    q: qkvBuf, qOffset: 0,
                    qOut: qScratch,
                    qInStride: convDim,
                    qOutStride: UInt32(cfg.numHeads * cfg.headDim),
                    k: kSlot.buffer, kOffset: kSlot.offset,
                    qWeight: qNorm.buffer, qWeightOffset: Int(qNorm.offset),
                    kWeight: kNorm.buffer, kWeightOffset: Int(kNorm.offset),
                    startPosition: UInt32(position),
                    t: 1,
                    headDim: UInt32(cfg.headDim),
                    numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(cfg.numFullKVHeads),
                    rotatedPairs: rotatedPairs,
                    theta: Float(cfg.fullRopeTheta),
                    eps: eps)
                attention.encodeFull(commandBuffer: cb,
                                     q: qScratch,
                                     k: kSlot.buffer, kOffset: 0,
                                     v: vSlot.buffer, vOffset: 0,
                                     out: attnOut,
                                     headDim: UInt32(cfg.headDim),
                                     numQHeads: UInt32(cfg.numHeads),
                                     numKVHeads: UInt32(cfg.numFullKVHeads),
                                     seqLen: seqLen,
                                     scale: attnScale)
                // Sigmoid output gate on the attention result (gate half of
                // the q projection, at element offset qRows/2 per token).
                laKernels.encodeAttnGate(
                    commandBuffer: cb,
                    attn: attnOut, attnOffset: 0,
                    gate: qkvBuf,
                    gateOffset: 0,
                    t: 1, d: qRows / 2,
                    headDim: UInt32(cfg.headDim),
                    gateStride: convDim)
                gemv(o, x: attnOut, y: oOut, m: D, n: qRows / 2)
            } else {
                let qkv = try model.linearInProjQKV(layer: L)
                let z = try model.linearInProjZ(layer: L)
                let a = try model.linearInProjA(layer: L)
                let b = try model.linearInProjB(layer: L)
                let conv = try model.linearConv1d(layer: L)
                let aLog = try model.linearALog(layer: L)
                let dtBias = try model.linearDtBias(layer: L)
                let outProj = try model.linearOutProj(layer: L)
                let normW = try model.linearNorm(layer: L)
                let stateBuf = try kv?.stateBuffer(layer: L)
                let convWindow = try kv?.convWindowBuffer(layer: L)
                guard let stateBuf, let convWindow else {
                    throw PrefillError.prefillCursorMismatch(
                        "linear layer \(L) is missing its state buffers")
                }

                func gemv(_ view: TensorView, x: MTLBuffer, y: MTLBuffer,
                          yOffset: Int = 0, m: UInt32, n: UInt32) {
                    int4.encode(commandBuffer: cb,
                                weights: view.buffer, weightsOffset: Int(view.offset),
                                scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                                biases: view.buffer, biasesOffset: Int(view.biasOffset),
                                x: x, y: y, yOffset: yOffset,
                                m: m, n: n)
                }
                gemv(qkv, x: normed, y: qkvBuf, m: convDim, n: D)
                gemv(z, x: normed, y: zBuf, m: nv * dv, n: D)
                gemv(a, x: normed, y: abBuf, yOffset: 0, m: nv, n: D)
                gemv(b, x: normed, y: abBuf, yOffset: Int(nv) * MemoryLayout<Float16>.stride,
                     m: nv, n: D)
                laKernels.encodeConv1dSilu(
                    commandBuffer: cb,
                    qkv: qkvBuf,
                    convWindow: convWindow,
                    weight: conv.buffer, weightOffset: Int(conv.offset),
                    convOut: convOutBuf,
                    t: 1, channels: convDim,
                    kernelSize: UInt32(cfg.linearConvKernelDim))
                laKernels.encodeConvWindowUpdate(
                    commandBuffer: cb,
                    qkv: qkvBuf,
                    oldWindow: convWindow,
                    newWindow: windowStageBuf,
                    t: 1, channels: convDim,
                    kernelSize: UInt32(cfg.linearConvKernelDim))
                if let blit = cb.makeBlitCommandEncoder() {
                    blit.copy(from: windowStageBuf, sourceOffset: 0,
                              to: convWindow, destinationOffset: 0,
                              size: convWindow.length)
                    blit.endEncoding()
                }
                laKernels.encodeL2NormQK(
                    commandBuffer: cb,
                    convOut: convOutBuf,
                    t: 1, numHeads: nk, headDim: dk,
                    rowStride: convDim,
                    eps: eps)
                laKernels.encodeGDNRecurse(
                    commandBuffer: cb,
                    q: convOutBuf, k: convOutBuf,
                    kOffset: Int(nk * dk) * MemoryLayout<Float16>.stride,
                    v: convOutBuf,
                    vOffset: Int(nk * dk * 2) * MemoryLayout<Float16>.stride,
                    a: abBuf, aOffset: 0,
                    b: abBuf, bOffset: Int(nv) * MemoryLayout<Float16>.stride,
                    aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                    dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                    state: stateBuf,
                    y: attnOut,
                    t: 1,
                    numKHeads: nk, numVHeads: nv,
                    keyHeadDim: dk, valueHeadDim: dv,
                    scalarStride: nv,
                    qkvStride: convDim,
                    yStride: nv * dv)
                laKernels.encodeGDNGatedNorm(
                    commandBuffer: cb,
                    y: attnOut,
                    z: zBuf,
                    weight: normW.buffer, weightOffset: Int(normW.offset),
                    out: attnOut,
                    t: 1, numVHeads: nv, valueHeadDim: dv,
                    eps: eps)
                gemv(outProj, x: attnOut, y: oOut, m: D, n: nv * dv)
            }

            // Common tail: residual + post-attention norm + router.
            laKernels.encodePostAttnSetup(
                commandBuffer: cb,
                h: hidden, hOffset: 0,
                r: oOut, rOffset: 0,
                normed2: normed,
                weight: postAttn.buffer, weightOffset: Int(postAttn.offset),
                t: 1, d: D,
                eps: eps)
            moe.encodeRouterGemma4(
                commandBuffer: cb,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                hidden: normed,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: routerPerExpertOnes,
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
            cb.commit()
            waitUntilCompleted(cb)
            try drainPending(wait: false)
            try checkCommandBufferError(cb)

            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }

            let routedOffsets = model.routedExpertOffsets(layer: L)
            let topK = UInt32(cfg.topKExperts)

            let sharedCB = ctx.queue.makeCommandBuffer()!
            sharedCB.label = "qwen decode L\(L) pos=\(position) shared"
            try! shared.encode(commandBuffer: sharedCB,
                               x: normed,
                               gate: sharedProj.gate,
                               up: sharedProj.up,
                               down: sharedProj.down,
                               y: h1Buf,
                               scratchGate: denseScratchGate,
                               scratchUp: denseScratchUp,
                               scratchAct: denseScratchAct)
            int8.encode(commandBuffer: sharedCB,
                        weights: sharedGateView.buffer,
                        weightsOffset: Int(sharedGateView.offset),
                        scales: sharedGateView.buffer,
                        scalesOffset: Int(sharedGateView.scaleOffset),
                        biases: sharedGateView.buffer,
                        biasesOffset: Int(sharedGateView.biasOffset),
                        x: normed,
                        y: gateLogitBuf,
                        m: 1, n: D)
            laKernels.encodeSharedGateMul(
                commandBuffer: sharedCB,
                sharedOut: h1Buf,
                gateLogit: gateLogitBuf,
                t: 1, d: D)
            sharedCB.commit()

            let blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
            let routedBufs = blobs.map { $0.buffer }
            let argBuf = moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs, topK: topK)
            let routedCB = ctx.queue.makeCommandBuffer()!
            routedCB.label = "qwen decode L\(L) pos=\(position) routed"
            moe.encodeRoutedPersistentPhase1U16Load(
                commandBuffer: routedCB,
                routedArgBuffer: argBuf,
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                x: normed,
                acts: moeActs,
                d: D, f: FmoE, topK: topK)
            moe.encodeRoutedPersistentPhase2Reduce(
                commandBuffer: routedCB,
                routedArgBuffer: argBuf,
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                acts: moeActs,
                routingWeights: outWeights,
                residual: zeroResidual,
                y: h2Buf,
                d: D, f: FmoE, topK: topK)
            laKernels.encodeLayerTail(
                commandBuffer: routedCB,
                hidden: hidden, hiddenOffset: 0,
                h1: h1Buf, h1Offset: 0,
                h2: h2Buf, h2Offset: 0,
                t: 1, d: D)
            routedCB.commit()
            precondition(pending == nil,
                         "qwen routed command-buffer pipeline drained before queuing")
            if let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                waitUntilCompleted(sharedCB)
                try checkCommandBufferError(sharedCB)
                waitUntilCompleted(routedCB)
                try checkCommandBufferError(routedCB)
                let data = Data(bytes: hidden.contents(), count: Int(D) * 2)
                try? data.write(to: URL(fileURLWithPath: "\(dumpDir)/dlayer_\(L).f16"))
                if L == 0 {
                    for (name, buf, count) in [("qkv", qkvBuf, Int(convDim) * 2),
                                               ("ab", abBuf, abBuf.length),
                                               ("z", zBuf, Int(nv * dv) * 2),
                                               ("convout", convOutBuf, Int(convDim) * 2),
                                               ("attnout", attnOut, Int(nv * dv) * 2),
                                               ("oout", oOut, Int(D) * 2),
                                               ("h1", h1Buf, Int(D) * 2),
                                               ("h2", h2Buf, Int(D) * 2),
                                               ("normed", normed, Int(D) * 2)] {
                        let d = Data(bytes: buf.contents(), count: count)
                        try? d.write(to: URL(fileURLWithPath: "\(dumpDir)/d0_\(name).f16"))
                    }
                    if let sb = try kv?.stateBuffer(layer: 0) {
                        let sdata = Data(bytes: sb.contents(), count: sb.length)
                        try? sdata.write(to: URL(fileURLWithPath: "\(dumpDir)/d0_state_after.f32"))
                    }
                    if let cw = try kv?.convWindowBuffer(layer: 0) {
                        let wdata = Data(bytes: cw.contents(), count: cw.length)
                        try? wdata.write(to: URL(fileURLWithPath: "\(dumpDir)/d0_window_after.f16"))
                    }
                }
            }
            pending = Pending(cb: routedCB, sharedCB: sharedCB)
        }
        try drainPending(wait: true)

        // Head: identical to the Gemma path (final norm + lm_head GEMV, or
        // the fused greedy head).
        let fNorm = model.finalNorm
        let lm = model.lmHead

        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            if useFusedHeadForThisToken {
                try runSync { cb in
                    self.fusionHead.encodeGreedyDecode(
                        commandBuffer: cb,
                        hidden: self.hidden,
                        normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                        weights: lm.buffer, weightsOffset: Int(lm.offset),
                        scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                        biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                        outToken: self.greedyTokenBuf,
                        d: D, vocab: UInt32(self.cfg.vocabSize),
                        rmsEps: eps)
                }
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                try runSync { cb in
                    self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                         weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                         out: self.normed, d: D, eps: eps)
                    self.int4.encode(commandBuffer: cb,
                                     weights: lm.buffer, weightsOffset: Int(lm.offset),
                                     scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                     biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                     x: self.normed, y: logits,
                                     m: UInt32(self.cfg.vocabSize), n: D)
                }
                if let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                    let n = Int(cfg.vocabSize)
                    let data = Data(bytes: logits.contents(), count: n * 2)
                    try? data.write(to: URL(fileURLWithPath: "\(dumpDir)/logits.f16"))
                }
            }
        }

        kv?.advance()
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        waitUntilCompleted(cb)
        try checkCommandBufferError(cb)
    }

    private nonisolated func waitUntilCompleted(_ cb: MTLCommandBuffer) {
        cb.waitUntilCompleted()
    }

}
