import Foundation

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if `nonFullLayerKind`. Indexed by layer.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String
    /// Set for architectures recorded explicitly in the manifest; nil keeps
    /// the legacy Gemma-shaped v1 manifests byte-identical.
    let archName: String?
    let nonFullLayerKind: String?

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tc = root["text_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no text_config")
        }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        if layerTypes.contains("linear_attention") {
            return try loadLinearHybrid(configPath: configPath, root: root, tc: tc, layerTypes: layerTypes)
        }
        return try loadSlidingFull(configPath: configPath, tc: tc)
    }

    /// Qwen3-Next-style hybrids (`qwen3_5_moe`): linear-attention layers use
    /// the gated delta rule, the rest are full-attention GQA. There is no
    /// sliding window, no softcap, and no separate global-head geometry.
    private static func loadLinearHybrid(configPath: String,
                                         root: [String: Any],
                                         tc: [String: Any],
                                         layerTypes: [String]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func s(_ k: String) throws -> String {
            guard let v = tc[k] as? String else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return v
        }
        let modelType = (tc["model_type"] as? String) ?? (root["model_type"] as? String) ?? ""
        guard modelType.hasPrefix("qwen3_5") else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "linear-attention hybrid family \(modelType) is not supported")
        }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let theta = (rope["rope_theta"] as? Double)
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let prf = (rope["partial_rotary_factor"] as? Double)
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let headDim = try i("head_dim")
        let kvHeads = try i("num_key_value_heads")
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        return ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("shared_expert_intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: kvHeads,
            numFullKVHeads: kvHeads,
            headDim: headDim,
            fullHeadDim: headDim,
            vocabSize: try i("vocab_size"),
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: theta,
            fullRopeTheta: theta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("num_experts_per_tok"),
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? false,
            attentionKEqV: false,
            fullAttentionLayerMask: mask,
            hiddenActivation: try s("hidden_act"),
            archName: "qwen3.6-35b-a3b",
            nonFullLayerKind: "linear_attention")
    }

    /// Gemma-style configs: sliding-window and full attention share a KV
    /// geometry, a softcap may be present, and rope parameters are split per
    /// attention kind.
    private static func loadSlidingFull(configPath: String, tc: [String: Any]) throws -> ArchInfo {
        func i(_ k: String) throws -> Int {
            guard let n = (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = (ropeFull["partial_rotary_factor"] as? Double)
            ?? (ropeFull["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let fullTheta = (ropeFull["rope_theta"] as? Double)
            ?? (ropeFull["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let swaTheta = (ropeSWA["rope_theta"] as? Double)
            ?? (ropeSWA["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000.0
        let kEqV = (tc["attention_k_eq_v"] as? Bool) ?? false
        let tie = (tc["tie_word_embeddings"] as? Bool) ?? false
        let act = (tc["hidden_activation"] as? String) ?? "gelu_pytorch_tanh"
        return ArchInfo(
            hiddenSize: try i("hidden_size"),
            intermediateSize: try i("intermediate_size"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            numFullKVHeads: try i("num_global_key_value_heads"),
            headDim: try i("head_dim"),
            fullHeadDim: try i("global_head_dim"),
            vocabSize: try i("vocab_size"),
            slidingWindow: try i("sliding_window"),
            finalLogitSoftcap: try d("final_logit_softcapping"),
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try i("num_hidden_layers"),
            numExperts: try i("num_experts"),
            topKExperts: try i("top_k_experts"),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            archName: nil,
            nonFullLayerKind: nil)
    }
}
