import Foundation
import Testing
@testable import TurboFieldfareFormat

private enum ArchFieldsFixture {
    static let zeroSHA = String(repeating: "0", count: 64)

    static func manifest(arch: GTurboManifestArchV1) -> GTurboManifestV1 {
        GTurboManifestV1(
            flags: [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false,
            ],
            modelID: "fixture/model",
            sourceSnapshotHash: "snapshot",
            arch: arch,
            quant: nil,
            files: [
                "model_weights.bin": GTurboManifestFileV1(size: 16_448, sha256: zeroSHA),
                "packed_experts/layout.json": GTurboManifestFileV1(size: 1, sha256: zeroSHA),
                "packed_experts/layer_00.bin": GTurboManifestFileV1(
                    size: 2 * GTurboFormatV1.alignmentBytes, sha256: zeroSHA),
            ],
            expertsPerLayer: 2,
            numLayers: 1,
            expertStride: GTurboFormatV1.alignmentBytes,
            bitWidthOverridesHonored: 0)
    }

    static func legacy() -> GTurboManifestV1 {
        manifest(arch: GTurboManifestArchV1(
            hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 1,
            headDim: 16, fullHeadDim: 32, vocabSize: 1024,
            slidingWindow: 128, finalLogitSoftcap: 30,
            ropeTheta: 10_000, fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25, numLayers: 1, numExperts: 2,
            topKExperts: 1, tieWordEmbeddings: true, attentionKEqV: true,
            hiddenActivation: "gelu_pytorch_tanh", fullAttentionLayerMask: [0]))
    }

    static func linearHybrid() -> GTurboManifestV1 {
        manifest(arch: GTurboManifestArchV1(
            hiddenSize: 64, ffnIntermediate: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 2,
            headDim: 32, fullHeadDim: 32, vocabSize: 1024,
            slidingWindow: 0, finalLogitSoftcap: 0,
            ropeTheta: 1_000_000, fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25, numLayers: 1, numExperts: 2,
            topKExperts: 1, tieWordEmbeddings: false, attentionKEqV: false,
            hiddenActivation: "silu", fullAttentionLayerMask: [0],
            archName: "qwen3.6-35b-a3b",
            nonFullLayerKind: GTurboManifestArchV1.linearAttentionKind))
    }

    /// Rebuilds a fixture arch with one field changed.
    static func withArch(
        from base: GTurboManifestArchV1,
        slidingWindow: Int? = nil,
        archName: String? = nil,
        nonFullLayerKind: String? = nil
    ) -> GTurboManifestArchV1 {
        GTurboManifestArchV1(
            hiddenSize: base.hiddenSize, ffnIntermediate: base.ffnIntermediate,
            moeIntermediateSize: base.moeIntermediateSize, numHeads: base.numHeads,
            numKVHeads: base.numKVHeads, numFullKVHeads: base.numFullKVHeads,
            headDim: base.headDim, fullHeadDim: base.fullHeadDim,
            vocabSize: base.vocabSize, slidingWindow: slidingWindow ?? base.slidingWindow,
            finalLogitSoftcap: base.finalLogitSoftcap, ropeTheta: base.ropeTheta,
            fullRopeTheta: base.fullRopeTheta,
            partialRotaryFactor: base.partialRotaryFactor,
            numLayers: base.numLayers, numExperts: base.numExperts,
            topKExperts: base.topKExperts, tieWordEmbeddings: base.tieWordEmbeddings,
            attentionKEqV: base.attentionKEqV, hiddenActivation: base.hiddenActivation,
            fullAttentionLayerMask: base.fullAttentionLayerMask,
            archName: archName ?? base.archName,
            nonFullLayerKind: nonFullLayerKind ?? base.nonFullLayerKind)
    }
}

@Suite
struct GTurboManifestArchFieldsTests {
    @Test func legacyManifestOmitsNewFieldsAndStillValidates() throws {
        let legacy = ArchFieldsFixture.legacy()
        let data = try GTurboManifestCodec.encode(legacy)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let arch = json["arch"] as! [String: Any]
        #expect(arch["archName"] == nil)
        #expect(arch["nonFullLayerKind"] == nil)
        #expect(try GTurboManifestCodec.decode(data) == legacy)
    }

    @Test func linearHybridManifestRoundTrips() throws {
        let hybrid = ArchFieldsFixture.linearHybrid()
        let data = try GTurboManifestCodec.encode(hybrid)
        #expect(try GTurboManifestCodec.decode(data) == hybrid)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let arch = json["arch"] as! [String: Any]
        #expect(arch["archName"] as? String == "qwen3.6-35b-a3b")
        #expect(arch["nonFullLayerKind"] as? String
            == GTurboManifestArchV1.linearAttentionKind)
    }

    @Test func unknownNonFullLayerKindIsRejected() throws {
        let manifest = ArchFieldsFixture.manifest(arch: ArchFieldsFixture.withArch(
            from: ArchFieldsFixture.legacy().arch,
            slidingWindow: 128,
            nonFullLayerKind: "gated_delta_rule"))
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.encode(manifest)
        }
    }

    @Test func linearAttentionKindAllowsZeroWindowButNotNegative() throws {
        // Zero is valid for linear-attention hybrids.
        _ = try GTurboManifestCodec.encode(ArchFieldsFixture.linearHybrid())

        let negative = ArchFieldsFixture.manifest(arch: ArchFieldsFixture.withArch(
            from: ArchFieldsFixture.linearHybrid().arch, slidingWindow: -1))
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.encode(negative)
        }
    }

    @Test func emptyArchNameIsRejected() throws {
        let manifest = ArchFieldsFixture.manifest(arch: ArchFieldsFixture.withArch(
            from: ArchFieldsFixture.linearHybrid().arch, archName: ""))
        #expect(throws: GTurboFormatError.self) {
            try GTurboManifestCodec.encode(manifest)
        }
    }
}
