import Foundation
import Darwin
import Metal

/// Which per-layer recurrent state a layer keeps. Gemma 4 interleaves
/// sliding-window and full-attention layers; Qwen3.6 hybrids run
/// gated-delta-rule linear-attention layers, which keep an fp32
/// `[numVHeads, Dv, Dk]` state and a conv window instead of K/V rows.
/// Sourced from `ArchConfig.fullAttentionLayerMask` plus
/// `ArchConfig.nonFullLayerKind`.
public enum LayerKind: Sendable { case swa, full, linear }

/// A read view the attention kernels bind. `offset` stays 0; ring-enabled SWA
/// layers expose the physical start slot for diagnostics while kernels map
/// logical positions with the supplied ring capacity.
public struct KVView: @unchecked Sendable {
    public let buffer: MTLBuffer
    /// Byte offset of logical position 0. Always 0 under linear storage.
    public let offset: Int
    /// Bytes per token (numKVHeads * headDim * sizeof(FP16)).
    public let stride: Int
    /// Number of valid positions written so far (== `position`). Attention reads
    /// `[0, validTokenCount]` (inclusive of the just-written token).
    public let validTokenCount: Int
    /// Ring start slot. 0 under linear storage; the hook for the wrap-aware path.
    public let startSlot: Int
}

/// Per-layer FP16 K/V storage for the decode loop.
///
/// One K buffer and one V buffer per layer, allocated once in `init` — the
/// decode hot path never allocates. Linear storage sizes every layer for
/// `maxContext`; FP16 ring storage caps SWA layers to their physical capacity
/// while full-attention layers remain linear.
///
/// Gemma 4 full-attention layers carry the `attention_k_eq_v` quirk: K and V
/// share the `k_proj` weight, so a single 4-bit dequant + GEMV produces the
/// raw projection. But after that the K-slot runs `k_norm` (per-head, with
/// scale) + RoPE while the V-slot runs `v_norm` (per-head, no scale) and
/// skips RoPE — they diverge before entering attention. So the cache buffers
/// must be separate; aliasing them would smash the V values with K's normed,
/// rotated bytes (gemma4-block.md §2.2). We always allocate K and V slots,
/// independent of `attentionKEqV`.
///
/// The K/V projection GEMV writes straight into the slot returned by
/// `kSlot`/`vSlot` (no separate `kv_write` kernel); the runner then norms +
/// optionally RoPE's each slot in place. `advance()` bumps the cursor once
/// both are written.
///
/// 8 GB rule: storage is bounded by per-layer physical capacity, allocated
/// once. `reset()` returns physical pages to the OS via `MADV_DONTNEED` so a
/// finished generation does not keep its KV resident into the next turn.
public final class KVCacheManager {
    public let config: ArchConfig
    public let maxContext: Int
    public let fp16RingEnabled: Bool

    private let kBuffers: [MTLBuffer]
    private let vBuffers: [MTLBuffer]
    private let strides:  [Int]         // bytes per token, per layer
    private let kinds:    [LayerKind]
    private let capacityTokens: [Int]
    private let slidingWindowTokens: Int

    public private(set) var position: Int = 0
    public private(set) var highWaterPosition: Int = 0

    private static let fp16Size = 2

    public init(device: MTLDevice,
                config: ArchConfig,
                maxContext: Int,
                fp16RingEnabled: Bool = false,
                slidingWindow: Int? = nil,
                maxPrefillChunkTokens: Int = 128,
                fp16RingCapacityOverride: Int? = nil) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(maxPrefillChunkTokens > 0, "maxPrefillChunkTokens must be positive")
        // Refused before a single `makeBuffer`: a context past the checkpoint's
        // position ceiling has no valid RoPE positions, and allocating 5+ GiB
        // of KV before saying so is the expensive way to learn it.
        guard maxContext <= config.maxPositionEmbeddings else {
            throw ModelError.contextExceedsModel(requested: maxContext,
                                                 maximum: config.maxPositionEmbeddings)
        }
        self.config = config
        self.maxContext = maxContext
        let ringEnabled = fp16RingEnabled
        self.fp16RingEnabled = ringEnabled

        let swaStride  = config.numKVHeads     * config.headDim     * Self.fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim  * Self.fp16Size
        let swaCapacity = min(maxContext,
                              max(1, fp16RingCapacityOverride
                                  ?? ((slidingWindow ?? config.slidingWindow) + maxPrefillChunkTokens)))
        self.slidingWindowTokens = slidingWindow ?? config.slidingWindow

        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        var st: [Int] = []
        var kd: [LayerKind] = []
        var caps: [Int] = []
        ks.reserveCapacity(config.numLayers)
        vs.reserveCapacity(config.numLayers)
        st.reserveCapacity(config.numLayers)
        kd.reserveCapacity(config.numLayers)
        caps.reserveCapacity(config.numLayers)

        for layer in 0..<config.numLayers {
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let kind: LayerKind
            let stride: Int
            let capacity: Int
            let kLength: Int
            let vLength: Int
            if config.usesLinearAttention && !isFull {
                // fp32 delta-rule state [numVHeads, Dv, Dk] in the K slot,
                // fp16 conv window [kernel - 1, convDim] in the V slot. No
                // per-token rows: stride 0, capacity 1.
                let stateBytes = config.linearNumValueHeads
                    * config.linearValueHeadDim
                    * config.linearKeyHeadDim
                    * MemoryLayout<Float32>.size
                let convBytes = (config.linearConvKernelDim - 1)
                    * (config.linearNumKeyHeads * config.linearKeyHeadDim * 2
                       + config.linearNumValueHeads * config.linearValueHeadDim)
                    * Self.fp16Size
                kind = .linear
                stride = 0
                capacity = 1
                kLength = stateBytes
                vLength = convBytes
            } else {
                kind = isFull ? .full : .swa
                stride = isFull ? fullStride : swaStride
                capacity = ringEnabled && !isFull ? swaCapacity : maxContext
                kLength = capacity * stride
                vLength = kLength
            }

            guard let kBuf = device.makeBuffer(length: max(kLength, 1), options: .storageModeShared) else {
                throw ModelError.kvAllocationFailed(layer: layer, bytes: kLength)
            }
            kBuf.label = "kv.K.layer\(layer)"
            // Metal recycles buffer pages without zeroing; the delta-rule
            // state and the conv window both start from zero, so linear
            // layers get an explicit clear (KV layers never read unwritten
            // rows and skip this).
            if kind == .linear {
                memset(kBuf.contents(), 0, kLength)
                if ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] != nil {
                    let base = kBuf.contents()
                    let v0 = base.load(as: Float32.self)
                    let v1 = base.load(fromByteOffset: 4, as: Float32.self)
                    let v2 = base.load(fromByteOffset: 8, as: Float32.self)
                    print("[kvinit] layer", layer, "state[0..2] =", v0, v1, v2)
                }
            }
            ks.append(kBuf)

            guard let vBuf = device.makeBuffer(length: max(vLength, 1), options: .storageModeShared) else {
                throw ModelError.kvAllocationFailed(layer: layer, bytes: vLength)
            }
            vBuf.label = "kv.V.layer\(layer)"
            if kind == .linear {
                memset(vBuf.contents(), 0, vLength)
            }
            vs.append(vBuf)

            st.append(stride)
            kd.append(kind)
            caps.append(capacity)
        }

        self.kBuffers = ks
        self.vBuffers = vs
        self.strides  = st
        self.kinds    = kd
        self.capacityTokens = caps
    }

    public func layerKind(_ layer: Int) -> LayerKind { kinds[layer] }

    /// Bytes per token for `layer` (K and V share the same stride).
    public func stride(layer: Int) -> Int { strides[layer] }

    /// Physical token capacity for `layer`. Ring-enabled SWA layers can be
    /// smaller than `maxContext`; full layers and ring-off storage stay linear.
    public func capacity(layer: Int) -> Int { capacityTokens[layer] }

    public func ringCapacity(layer: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        return capacityTokens[layer]
    }

    /// Total bytes of the K buffer for `layer`.
    public func bufferLength(layer: Int) -> Int {
        return capacityTokens[layer] * strides[layer]
    }

    /// Read/write buffer for a linear-attention layer's fp32 recurrence
    /// state `[numVHeads, Dv, Dk]`. `nil` unless the layer is `.linear`.
    public func stateBuffer(layer: Int) -> MTLBuffer? {
        guard kinds[layer] == .linear else { return nil }
        return kBuffers[layer]
    }

    /// Read/write buffer for a linear-attention layer's depthwise conv
    /// window `[kernel - 1, convDim]` FP16. `nil` unless the layer is
    /// `.linear`.
    public func convWindowBuffer(layer: Int) -> MTLBuffer? {
        guard kinds[layer] == .linear else { return nil }
        return vBuffers[layer]
    }

    /// Write target for this layer's K projection at `position`. Linear
    /// layers have no per-token rows and must not use this.
    public func kSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        precondition(kinds[layer] != .linear,
                     "kSlot is not valid for a linear-attention state layer")
        validateRange(start: position, count: 1)
        return (kBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    /// Write target for this layer's V projection at `position`. Linear layers
    /// have no per-token rows and must not use this.
    public func vSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        precondition(kinds[layer] != .linear,
                     "vSlot is not valid for a linear-attention state layer")
        validateRange(start: position, count: 1)
        return (vBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    public func kRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (kBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func vRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (vBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func keyView(layer: Int) -> KVView {
        keyView(layer: layer, validTokenCount: position)
    }

    public func keyView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return KVView(buffer: kBuffers[layer], offset: 0, stride: strides[layer],
                      validTokenCount: validTokenCount, startSlot: ringStartSlot(layer: layer,
                                                                                 validTokenCount: validTokenCount))
    }

    public func valueView(layer: Int) -> KVView {
        valueView(layer: layer, validTokenCount: position)
    }

    func keyBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        keyView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    func valueBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        valueView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    public func valueView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return KVView(buffer: vBuffers[layer], offset: 0, stride: strides[layer],
                      validTokenCount: validTokenCount, startSlot: ringStartSlot(layer: layer,
                                                                                 validTokenCount: validTokenCount))
    }

    /// Advance the position cursor once the current token's K/V are written
    /// across all layers.
    public func advance() { advance(by: 1) }

    public func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= maxContext, "advance would exceed maxContext")
        position += count
        highWaterPosition = max(highWaterPosition, position)
    }

    /// Maximum safe rewind before a reused SWA ring slot could become visible.
    /// Delta-rule state cannot be un-applied, so any arch with linear-attention
    /// layers refuses to rewind at all (server continuation must reset).
    public var maxRewindTokens: Int {
        if kinds.contains(.linear) { return 0 }
        guard fp16RingEnabled else { return maxContext }
        var slack = maxContext
        for layer in 0..<config.numLayers where kinds[layer] == .swa {
            let capacity = capacityTokens[layer]
            guard capacity < maxContext else { continue }
            slack = min(slack, max(0, capacity - slidingWindowTokens))
        }
        return slack
    }

    /// Abandon the newest rows while keeping the remaining KV lineage intact.
    public func rewind(to newPosition: Int) {
        precondition(newPosition >= 0, "rewind target must be non-negative")
        precondition(newPosition <= position,
                     "rewind target \(newPosition) is beyond position \(position)")
        precondition(highWaterPosition - newPosition <= maxRewindTokens,
                     "rewind to \(newPosition) exceeds ring slack \(maxRewindTokens)")
        position = newPosition
    }

    /// Drop all cached positions and advise the pages away.
    ///
    /// No buffer zeroing — the attention kernels read only `[0, validTokenCount]`,
    /// and `validTokenCount` is now 0. The `MADV_DONTNEED` is advisory only:
    /// measured on macOS 26 (M5 Pro, 2026-08-28), `phys_footprint` did not move
    /// after advising a fully written 262,144-row cache, and a single attention
    /// bind had already charged each bound buffer in full. Sizing decisions
    /// must therefore treat the KV as resident at its allocated size for as
    /// long as the runner lives; see `ContextAdmission`.
    public func reset() {
        position = 0
        highWaterPosition = 0
        let pageSize = Int(getpagesize())
        var advised = Set<ObjectIdentifier>()
        for layer in 0..<config.numLayers {
            // Delta-rule state cannot survive a reset: zero it back to the
            // fresh-generation state (the conv window with it).
            if kinds[layer] == .linear {
                memset(kBuffers[layer].contents(), 0, kBuffers[layer].length)
                memset(vBuffers[layer].contents(), 0, vBuffers[layer].length)
            }
            advise(kBuffers[layer], pageSize: pageSize, seen: &advised)
            advise(vBuffers[layer], pageSize: pageSize, seen: &advised)
        }
    }

    private func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(start + count <= maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    private func validateValidTokenCount(_ count: Int) {
        precondition(count >= 0, "validTokenCount must be non-negative")
        precondition(count <= maxContext,
                     "validTokenCount \(count) exceeds maxContext \(maxContext)")
    }

    private func physicalSlot(layer: Int, position: Int) -> Int {
        position % capacityTokens[layer]
    }

    private func ringStartSlot(layer: Int, validTokenCount: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        let capacity = capacityTokens[layer]
        guard validTokenCount > capacity else { return 0 }
        return validTokenCount % capacity
    }

    private func validateContiguousPhysicalRange(layer: Int, start: Int, count: Int) {
        guard count > 0, fp16RingEnabled, kinds[layer] == .swa else { return }
        let capacity = capacityTokens[layer]
        let physicalStart = start % capacity
        precondition(physicalStart + count <= capacity,
                     "range \(start)..<\(start + count) wraps FP16 KV ring capacity \(capacity)")
    }

    private func advise(_ buffer: MTLBuffer, pageSize: Int, seen: inout Set<ObjectIdentifier>) {
        let id = ObjectIdentifier(buffer)
        if seen.contains(id) { return }
        seen.insert(id)
        // MTLBuffer allocations are page-aligned; round the length down to a
        // whole number of pages so we never hand madvise a partial tail page.
        let len = (buffer.length / pageSize) * pageSize
        if len > 0 {
            _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED)
        }
    }
}
