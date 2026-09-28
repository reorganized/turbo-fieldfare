import Foundation

/// Synthesises a Qwen3.6-shaped hybrid snapshot for the repack planner tests:
/// 40 layers x 256 experts with `.mlp.switch_mlp` routed experts, linear
/// attention layers, an 8-bit `mlp.gate` router, and an 8-bit
/// `mlp.shared_expert_gate`. Shapes are tiny but structurally faithful to
/// `mlx-community/Qwen3.6-35B-A3B-4bit`.
enum QwenSyntheticSnapshot {

    struct Arch {
        static let hidden = 128
        static let sharedIntermediate = 128
        static let moeIntermediate = 64
        static let numHeads = 4
        static let kvHeads = 2
        static let headDim = 32
        static let linearHeads = 2
        static let linearHeadDim = 32
        static let vocab = 256
        static let numLayers = 40
        static let numExperts = 256
        static let topK = 8
        static let groupSize = 64
        static let convKernel = 4
        /// Layers 0-2 full, then every 4th full.
        static func isFull(_ layer: Int) -> Bool {
            layer < 3 || layer % 4 == 0
        }
    }

    static func build(at dir: String, seed: UInt64 = 0x51DE_29A4_7C01_BEEF) throws {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        var rng = SplitMix64(seed: seed)
        var tensors: [(String, String, [Int], [UInt8])] = []

        appendQuantizedWeight(name: "language_model.model.embed_tokens",
                              outerShape: [Arch.vocab],
                              innerLogical: Arch.hidden, bits: 4,
                              groupSize: Arch.groupSize, into: &tensors, rng: &rng)
        appendQuantizedWeight(name: "language_model.lm_head",
                              outerShape: [Arch.vocab],
                              innerLogical: Arch.hidden, bits: 4,
                              groupSize: Arch.groupSize, into: &tensors, rng: &rng)

        let linearOut = Arch.linearHeads * Arch.linearHeadDim
        let qOut = Arch.numHeads * Arch.headDim
        let kvOut = Arch.kvHeads * Arch.headDim

        for li in 0..<Arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            let full = Arch.isFull(li)

            if full {
                appendQuantizedWeight(name: prefix + ".self_attn.q_proj",
                                      outerShape: [qOut], innerLogical: Arch.hidden,
                                      bits: 4, groupSize: Arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.k_proj",
                                      outerShape: [kvOut], innerLogical: Arch.hidden,
                                      bits: 4, groupSize: Arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.v_proj",
                                      outerShape: [kvOut], innerLogical: Arch.hidden,
                                      bits: 4, groupSize: Arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".self_attn.o_proj",
                                      outerShape: [Arch.hidden], innerLogical: qOut,
                                      bits: 4, groupSize: Arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.q_norm.weight",
                                      shape: [Arch.headDim], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".self_attn.k_norm.weight",
                                      shape: [Arch.headDim], into: &tensors, rng: &rng)
            } else {
                // Gated delta-rule linear attention: fused qkv, sigmoid z gate,
                // low-rank dt/beta, depthwise conv, out projection. Channel
                // counts reuse the attention head geometry; the planner is
                // shape-agnostic for resident tensors, so exact head math
                // does not matter here.
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_qkv",
                                      outerShape: [qOut + 2 * kvOut],
                                      innerLogical: Arch.hidden, bits: 4,
                                      groupSize: Arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_z",
                                      outerShape: [linearOut],
                                      innerLogical: Arch.hidden, bits: 4,
                                      groupSize: Arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_a",
                                      outerShape: [Arch.linearHeads],
                                      innerLogical: Arch.hidden, bits: 4,
                                      groupSize: Arch.groupSize, into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.in_proj_b",
                                      outerShape: [Arch.linearHeads],
                                      innerLogical: Arch.hidden, bits: 4,
                                      groupSize: Arch.groupSize, into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.conv1d.weight",
                                      shape: [qOut + 2 * kvOut, Arch.convKernel, 1],
                                      into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.A_log",
                                      shape: [Arch.linearHeads], into: &tensors, rng: &rng)
                appendUnquantizedBF16(name: prefix + ".linear_attn.dt_bias",
                                      shape: [Arch.linearHeads], into: &tensors, rng: &rng)
                appendQuantizedWeight(name: prefix + ".linear_attn.out_proj",
                                      outerShape: [Arch.hidden], innerLogical: linearOut,
                                      bits: 4, groupSize: Arch.groupSize, into: &tensors, rng: &rng)
            }

            // Router — 8-bit.
            appendQuantizedWeight(name: prefix + ".mlp.gate",
                                  outerShape: [Arch.numExperts], innerLogical: Arch.hidden,
                                  bits: 8, groupSize: Arch.groupSize, into: &tensors, rng: &rng)
            // Shared expert — 4-bit, plus an 8-bit scalar gate.
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert.gate_proj",
                                  outerShape: [Arch.sharedIntermediate], innerLogical: Arch.hidden,
                                  bits: 4, groupSize: Arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert.up_proj",
                                  outerShape: [Arch.sharedIntermediate], innerLogical: Arch.hidden,
                                  bits: 4, groupSize: Arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert.down_proj",
                                  outerShape: [Arch.hidden], innerLogical: Arch.sharedIntermediate,
                                  bits: 4, groupSize: Arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.shared_expert_gate",
                                  outerShape: [1], innerLogical: Arch.hidden,
                                  bits: 8, groupSize: Arch.groupSize, into: &tensors, rng: &rng)

            // Routed experts — 4-bit, pre-stacked rank-3.
            appendQuantizedWeight(name: prefix + ".mlp.switch_mlp.gate_proj",
                                  outerShape: [Arch.numExperts, Arch.moeIntermediate],
                                  innerLogical: Arch.hidden, bits: 4,
                                  groupSize: Arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.switch_mlp.up_proj",
                                  outerShape: [Arch.numExperts, Arch.moeIntermediate],
                                  innerLogical: Arch.hidden, bits: 4,
                                  groupSize: Arch.groupSize, into: &tensors, rng: &rng)
            appendQuantizedWeight(name: prefix + ".mlp.switch_mlp.down_proj",
                                  outerShape: [Arch.numExperts, Arch.hidden],
                                  innerLogical: Arch.moeIntermediate, bits: 4,
                                  groupSize: Arch.groupSize, into: &tensors, rng: &rng)

            appendUnquantizedBF16(name: prefix + ".input_layernorm.weight",
                                  shape: [Arch.hidden], into: &tensors, rng: &rng)
            appendUnquantizedBF16(name: prefix + ".post_attention_layernorm.weight",
                                  shape: [Arch.hidden], into: &tensors, rng: &rng)
        }
        appendUnquantizedBF16(name: "language_model.model.norm.weight",
                              shape: [Arch.hidden], into: &tensors, rng: &rng)

        let shardName = "model-00001-of-00001.safetensors"
        let shardPath = (dir as NSString).appendingPathComponent(shardName)
        try writeShard(path: shardPath, tensors: tensors)

        var overrides: [String: [String: Any]] = [:]
        for li in 0..<Arch.numLayers {
            let prefix = "language_model.model.layers.\(li)"
            overrides[prefix + ".mlp.gate"] = ["bits": 8, "group_size": Arch.groupSize]
            overrides[prefix + ".mlp.shared_expert_gate"] = ["bits": 8, "group_size": Arch.groupSize]
        }
        var quant: [String: Any] = [
            "bits": 4, "group_size": Arch.groupSize, "mode": "affine"
        ]
        for (k, v) in overrides { quant[k] = v }

        var layerTypes: [String] = []
        for li in 0..<Arch.numLayers {
            layerTypes.append(Arch.isFull(li) ? "full_attention" : "linear_attention")
        }
        let textConfig: [String: Any] = [
            "model_type": "qwen3_5_moe",
            "hidden_size": Arch.hidden,
            "shared_expert_intermediate_size": Arch.sharedIntermediate,
            "moe_intermediate_size": Arch.moeIntermediate,
            "num_attention_heads": Arch.numHeads,
            "num_key_value_heads": Arch.kvHeads,
            "head_dim": Arch.headDim,
            "vocab_size": Arch.vocab,
            "num_hidden_layers": Arch.numLayers,
            "num_experts": Arch.numExperts,
            "num_experts_per_tok": Arch.topK,
            "num_v_heads": Arch.linearHeads,
            "linear_key_head_dim": Arch.linearHeadDim,
            "linear_value_head_dim": Arch.linearHeadDim,
            "conv_kernel": Arch.convKernel,
            "hidden_act": "silu",
            "rope_parameters": [
                "rope_theta": 1_000_000.0,
                "partial_rotary_factor": 0.25
            ],
            "layer_types": layerTypes,
            "tie_word_embeddings": false
        ]
        let config: [String: Any] = [
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "model_type": "qwen3_5_moe",
            "quantization": quant,
            "text_config": textConfig
        ]
        let configData = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try configData.write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))

        var weightMap: [String: String] = [:]
        for (name, _, _, _) in tensors { weightMap[name] = shardName }
        let indexObj: [String: Any] = [
            "metadata": ["format": "mlx"],
            "weight_map": weightMap
        ]
        let indexData = try JSONSerialization.data(withJSONObject: indexObj, options: [.sortedKeys])
        let indexPath = (dir as NSString).appendingPathComponent("model.safetensors.index.json")
        try indexData.write(to: URL(fileURLWithPath: indexPath))
    }

    // MARK: - Tensor builders (same conventions as SyntheticSnapshot)

    private static func appendQuantizedWeight(name: String,
                                              outerShape: [Int],
                                              innerLogical: Int,
                                              bits: Int,
                                              groupSize: Int,
                                              into tensors: inout [(String, String, [Int], [UInt8])],
                                              rng: inout SplitMix64) {
        precondition(innerLogical % groupSize == 0)
        let factor = 32 / bits
        precondition(innerLogical % factor == 0)
        let innerSource = innerLogical / factor
        let shape = outerShape + [innerSource]
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 4)
        for i in 0..<bytes.count { bytes[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name + ".weight", "U32", shape, bytes))

        let groups = innerLogical / groupSize
        let companionShape = outerShape + [groups]
        let companionElems = companionShape.reduce(1, *)
        var sb = [UInt8](repeating: 0, count: companionElems * 2)
        for i in 0..<sb.count { sb[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name + ".scales", "BF16", companionShape, sb))
        var bb = [UInt8](repeating: 0, count: companionElems * 2)
        for i in 0..<bb.count { bb[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name + ".biases", "BF16", companionShape, bb))
    }

    private static func appendUnquantizedBF16(name: String, shape: [Int],
                                              into tensors: inout [(String, String, [Int], [UInt8])],
                                              rng: inout SplitMix64) {
        let elements = shape.reduce(1, *)
        var bytes = [UInt8](repeating: 0, count: elements * 2)
        for i in 0..<bytes.count { bytes[i] = UInt8(rng.next() & 0xFF) }
        tensors.append((name, "BF16", shape, bytes))
    }

    private static func writeShard(path: String,
                                   tensors: [(String, String, [Int], [UInt8])]) throws {
        var off: UInt64 = 0
        var headerDict: [String: Any] = [:]
        for (name, dtype, shape, bytes) in tensors {
            let begin = off
            let end = begin + UInt64(bytes.count)
            headerDict[name] = [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, end]
            ]
            off = end
        }
        headerDict["__metadata__"] = ["format": "mlx"]
        let headerData = try JSONSerialization.data(withJSONObject: headerDict,
                                                    options: [.sortedKeys])
        var padded = headerData
        while padded.count % 8 != 0 { padded.append(0x20) }

        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        precondition(fd >= 0, "open failed for \(path)")
        defer { close(fd) }
        var headerLenLE = UInt64(padded.count).littleEndian
        withUnsafeBytes(of: &headerLenLE) { raw in
            _ = write(fd, raw.baseAddress, 8)
        }
        padded.withUnsafeBytes { raw in
            _ = write(fd, raw.baseAddress, padded.count)
        }
        for (_, _, _, bytes) in tensors {
            bytes.withUnsafeBufferPointer { ptr in
                _ = write(fd, ptr.baseAddress, ptr.count)
            }
        }
    }
}
