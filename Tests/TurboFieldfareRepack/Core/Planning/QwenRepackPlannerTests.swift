import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite
struct QwenRepackPlannerTests {

    @Test func classifyRoutesSwitchMlpTensors() {
        let gate = "language_model.model.layers.5.mlp.switch_mlp.gate_proj.weight"
        let up = "language_model.model.layers.12.mlp.switch_mlp.up_proj.scales"
        let down = "language_model.model.layers.39.mlp.switch_mlp.down_proj.weight"
        #expect(RepackPlanner.classify(gate, numLayers: 40)
            == .routedExpert(role: "gate", layer: 5))
        #expect(RepackPlanner.classify(up, numLayers: 40)
            == .routedExpert(role: "up", layer: 12))
        #expect(RepackPlanner.classify(down, numLayers: 40)
            == .routedExpert(role: "down", layer: 39))
        // The Gemma pattern keeps classifying too.
        #expect(RepackPlanner.classify(
            "language_model.model.layers.0.experts.switch_glu.gate_proj.weight",
            numLayers: 1) == .routedExpert(role: "gate", layer: 0))
        // Router and shared-expert tensors stay resident.
        #expect(RepackPlanner.classify(
            "language_model.model.layers.0.mlp.gate.weight", numLayers: 40) == .lmResident)
        #expect(RepackPlanner.classify(
            "language_model.model.layers.0.mlp.shared_expert_gate.weight",
                numLayers: 40) == .lmResident)
    }

    @Test func plannerPlansQwenHybridSnapshot() throws {
        let snapshotDirectory = temporaryRoot("qwen-snapshot")
        let outputDirectory = temporaryRoot("qwen-out")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDirectory)
            try? FileManager.default.removeItem(atPath: outputDirectory)
        }
        try QwenSyntheticSnapshot.build(at: snapshotDirectory)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDirectory)
        let arch = try ArchInfo.load(
            configPath: (snapshotDirectory as NSString).appendingPathComponent("config.json"))
        let header = try parseShardHeader(
            path: (snapshotDirectory as NSString)
                .appendingPathComponent("model-00001-of-00001.safetensors"))

        #expect(arch.archName == "qwen3.6-35b-a3b")
        #expect(arch.nonFullLayerKind == "linear_attention")
        #expect(arch.slidingWindow == 0)
        #expect(arch.finalLogitSoftcap == 0)
        #expect(arch.numLayers == 40)
        #expect(arch.numExperts == 256)
        #expect(arch.fullAttentionLayerMask.count == 40)
        #expect(arch.fullAttentionLayerMask[0] == 1)
        #expect(arch.fullAttentionLayerMask[3] == 0)
        #expect(arch.fullAttentionLayerMask[4] == 1)
        #expect(arch.hiddenActivation == "silu")
        #expect(arch.attentionKEqV == false)

        let plan = try RepackPlanner.plan(
            meta: metadata, arch: arch, shardHeaders: [header],
            outputDir: outputDirectory)

        // 40 layers, 256 experts each, all three roles from `.mlp.switch_mlp`.
        #expect(plan.layers.count == 40)
        #expect(plan.bitsOverrideCount == 80)  // 40 x (mlp.gate + shared_expert_gate)
        for lp in plan.layers {
            #expect(lp.expertsPerLayer == 256)
            let roles = Dictionary(uniqueKeysWithValues:
                lp.subTensors.filter { $0.component == "weights" }
                    .map { ($0.role, $0) })
            #expect(Set(roles.keys) == ["gate", "up", "down"])
            for (role, sub) in roles {
                #expect(sub.sourceTensor.name
                    .contains(".mlp.switch_mlp.\(role)_proj.weight"))
                #expect(sub.bitsForWeights == 4)
                #expect(sub.sourceTensor.dtype == .u32)
                #expect(Int(sub.sourceTensor.shape[0]) == 256)
            }
        }

        // Router and shared-expert gate ride 8-bit overrides; the shared
        // expert projections stay 4-bit.
        let entries = Dictionary(uniqueKeysWithValues:
            plan.resident.entries.map { ($0.name, $0) })
        let layer0 = "language_model.model.layers.0."
        #expect(IndexLoader.quantSpec(
            forTensor: layer0 + "mlp.gate.weight", meta: metadata).bits == 8)
        #expect(IndexLoader.quantSpec(
            forTensor: layer0 + "mlp.shared_expert_gate.weight", meta: metadata).bits == 8)
        #expect(IndexLoader.quantSpec(
            forTensor: layer0 + ".mlp.shared_expert.gate_proj.weight",
            meta: metadata).bits == 4)
        #expect(entries[layer0 + "mlp.shared_expert_gate.weight"]!.quantSpec!.bits == 8)
        #expect(entries[layer0 + "mlp.shared_expert.down_proj.weight"]!.quantSpec!.bits == 4)
        #expect(entries["language_model.model.layers.39.linear_attn.in_proj_z.weight"] != nil)
        // lm_head is untied and must be present in the resident plan.
        #expect(entries["language_model.lm_head.weight"] != nil)
        #expect(plan.resident.entries.contains {
            $0.name == "language_model.model.layers.0.self_attn.q_proj.weight"
        })
        #expect(plan.matchedModelID == nil)
    }

    @Test func archInfoRejectsUnknownHybridFamily() throws {
        let dir = temporaryRoot("qwen-unknown")
        defer { try? FileManager.default.removeItem(atPath: dir) }
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)
        let config = """
        {"model_type": "olmo_hybrid", "text_config": {"layer_types": ["linear_attention"]}}
        """
        let path = (dir as NSString).appendingPathComponent("config.json")
        try Data(config.utf8).write(to: URL(fileURLWithPath: path))
        #expect(throws: RepackError.self) {
            try ArchInfo.load(configPath: path)
        }
    }

    // MARK: - Helpers

    private func temporaryRoot(_ tag: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("turbofieldfare-qwen-\(tag)-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true)
        return path
    }

    private func parseShardHeader(path: String) throws -> Safetensors.Header {
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        var headerSize: UInt64 = 0
        try withUnsafeMutableBytes(of: &headerSize) {
            try Posix.preadAll(
                fd: fd, path: path, buf: $0.baseAddress!, count: 8, offset: 0)
        }
        headerSize = UInt64(littleEndian: headerSize)
        var headerData = Data(count: Int(headerSize))
        try headerData.withUnsafeMutableBytes {
            try Posix.preadAll(
                fd: fd, path: path, buf: $0.baseAddress!,
                count: $0.count, offset: 8)
        }
        return try Safetensors.parseHeaderBytes(
            path: path,
            fileSize: try Posix.fileSize(fd: fd, path: path),
            headerBytes: headerData)
    }
}
