import Foundation
import Metal

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String
    /// Identity of the architecture this configuration describes. The
    /// manifest records the same value; loaders select the expected
    /// configuration from it.
    public let archName: String
    /// What the mask's 0-valued layers are: `sliding_attention` (Gemma) or
    /// `linear_attention` (recurrent, no KV).
    public let nonFullLayerKind: String
    /// The checkpoint's `max_position_embeddings`. The one ceiling every
    /// surface refuses above; not a manifest-compared field, so
    /// `ManifestReader.validateArch` does not read it.
    public let maxPositionEmbeddings: Int
    /// Gated-delta-rule geometry (0 when the arch has no linear-attention
    /// layers). Not manifest-compared; the manifest only needs the layer
    /// mask plus `nonFullLayerKind`.
    public let linearNumKeyHeads: Int
    public let linearNumValueHeads: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let linearConvKernelDim: Int

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String,
        archName: String = "gemma4-26b-a4b",
        nonFullLayerKind: String = "sliding_attention",
        maxPositionEmbeddings: Int = 262_144,
        linearNumKeyHeads: Int = 0,
        linearNumValueHeads: Int = 0,
        linearKeyHeadDim: Int = 0,
        linearValueHeadDim: Int = 0,
        linearConvKernelDim: Int = 0
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.archName = archName
        self.nonFullLayerKind = nonFullLayerKind
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.linearNumKeyHeads = linearNumKeyHeads
        self.linearNumValueHeads = linearNumValueHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelDim = linearConvKernelDim
    }

    /// True when the mask's 0-valued layers run the gated delta rule instead
    /// of sliding-window attention.
    public var usesLinearAttention: Bool {
        nonFullLayerKind == "linear_attention"
    }

    /// Attention softmax scale. Gemma 4 uses scale 1.0 (logit softcapping
    /// replaces the 1/sqrt(d) factor); Qwen3.6 uses the standard
    /// head_dim^-0.5.
    public var attentionScale: Float {
        usesLinearAttention
            ? Float(1.0 / (Double(headDim).squareRoot()))
            : 1.0
    }

    /// Embedding output scale. Gemma 4 multiplies the embedding by
    /// sqrt(hidden); Qwen3.6 leaves it at 1.0.
    public var embedOutScale: Float {
        usesLinearAttention ? 1.0 : Float(hiddenSize).squareRoot()
    }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh",
        maxPositionEmbeddings: 262_144
    )

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }

    /// Qwen3.6 35B-A3B (Qwen3-Next hybrid). 40 layers: 30 gated-delta-rule
    /// linear-attention layers and 10 full-attention GQA layers. Every layer
    /// is MoE (256 experts, top-8, softmax router with renormalization) plus
    /// one shared expert behind a sigmoid gate. The full-attention layers
    /// carry a real V projection (no K==V sharing) and fuse an output gate
    /// into q_proj. Head, embeddings, and rope thetas come from the pinned
    /// `mlx-community/Qwen3.6-35B-A3B-4bit` checkpoint.
    public static let qwen3_6_35B_A3B = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 40,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwen36LayerMask(),
        hiddenActivation: "silu",
        archName: "qwen3.6-35b-a3b",
        nonFullLayerKind: "linear_attention",
        maxPositionEmbeddings: 262_144,
        linearNumKeyHeads: 16,
        linearNumValueHeads: 32,
        linearKeyHeadDim: 128,
        linearValueHeadDim: 128,
        linearConvKernelDim: 4
    )

    private static func qwen36LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { mask[i] = 1 }
        return mask
    }

    /// Resolve the expected configuration from a manifest's
    /// `arch.archName`. Legacy (nil) manifests keep the caller's default —
    /// today that is only the Gemma baseline. A recorded name must map to a
    /// known profile; unknown names fail closed.
    public static func resolve(manifestArchName: String?,
                               fallback: ArchConfig) throws -> ArchConfig {
        guard let name = manifestArchName, !name.isEmpty else { return fallback }
        switch name {
        case gemma4_26B_A4B.archName: return gemma4_26B_A4B
        case qwen3_6_35B_A3B.archName: return qwen3_6_35B_A3B
        default:
            throw ModelError.indexCorrupt(
                detail: "manifest.arch.archName \"\(name)\" has no runtime profile")
        }
    }
}

/// Branch activation for the expert FFNs. Gemma 4 uses tanh-approximate
/// GELU; Qwen3-Next uses SiLU.
public enum GatedActivation: Sendable, Equatable {
    case geluPytorchTanh
    case silu

    public init(hiddenActivation: String) {
        self = hiddenActivation == "silu" ? .silu : .geluPytorchTanh
    }
}

/// Failure modes for the validation gates in `Model.load`.
public enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)
    case contextExceedsModel(requested: Int, maximum: Int)
    case kvAllocationFailed(layer: Int, bytes: Int)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        case .contextExceedsModel(let requested, let maximum):
            return "maxContext \(requested) exceeds the model's maximum position \(maximum)"
        case .kvAllocationFailed(let layer, let bytes):
            return "KV cache allocation of \(bytes) bytes for layer \(layer) failed"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
