import Foundation
import Metal
import Testing
@testable import TurboFieldfare

/// Differential tests for the Qwen3.6 hybrid kernels in `linear_attn.metal`
/// against straightforward Float32 CPU references. Runs anywhere a Metal
/// device exists; the references mirror mlx-lm's `gated_delta.py` and
/// `qwen3_5.py` math (fp32 state, scalar per-v-head gating).
@Suite
struct LinearAttnKernelTests {

    // MARK: - Helpers

    private struct Rng {
        var state: UInt64
        init(seed: UInt64) { self.state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func unit() -> Float {
            Float(next() >> 40) / Float(1 << 24) - 0.5
        }
        mutating func bf16() -> UInt16 {
            // Deterministic small-magnitude BF16 weights.
            let v = unit() * 0.25
            return LinearAttnKernelTests.bf16Bits(v)
        }
    }

    private static func bf16Bits(_ v: Float) -> UInt16 {
        let rounded = (v < 0 ? -v : v)
        var bits = v.bitPattern
        let lsb = (bits >> 16) & 1
        bits += 0x7FFF + lsb
        var out = UInt16(bits >> 16)
        if rounded.isNaN { out = 0x7FC0 }
        return out
    }

    private static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    private static func makeFP16(_ values: [Float], device: MTLDevice) -> MTLBuffer? {
        let halves = values.map { Float16($0) }
        return device.makeBuffer(bytes: halves,
                                 length: halves.count * MemoryLayout<Float16>.stride,
                                 options: .storageModeShared)
    }

    private static func readFP16(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(ptr[$0]) }
    }

    private static func writeFP16(_ buffer: MTLBuffer, _ values: [Float]) {
        let halves = values.map { Float16($0) }
        memcpy(buffer.contents(), halves, halves.count * MemoryLayout<Float16>.stride)
    }

    private static func writeBF16(_ buffer: MTLBuffer, _ bits: [UInt16]) {
        memcpy(buffer.contents(), bits, bits.count * MemoryLayout<UInt16>.size)
    }

    // MARK: - conv1d + SiLU

    @Test func conv1dSilu_matchesCPUReference() throws {
        let T = 7, C = 12, K = 4
        let ctx = try MetalContext()
        let kernels = try LinearAttnKernels(context: ctx)
        var rng = Rng(seed: 0xC0FFEE)

        let qkv = (0..<(T * C)).map { _ in rng.unit() }
        let windowRows = K - 1
        let window = (0..<(windowRows * C)).map { _ in rng.unit() }
        let weightBits = (0..<(C * K)).map { _ in rng.bf16() }

        guard let qkvBuf = Self.makeFP16(qkv, device: ctx.device),
              let winBuf = Self.makeFP16(window, device: ctx.device),
              let stageBuf = Self.makeFP16(Array(repeating: 0, count: windowRows * C),
                                      device: ctx.device),
              let outBuf = ctx.device.makeBuffer(
                length: T * C * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let weightBuf = ctx.device.makeBuffer(
                bytes: weightBits,
                length: weightBits.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernels.encodeConv1dSilu(commandBuffer: cb,
                                 qkv: qkvBuf,
                                 convWindow: winBuf,
                                 weight: weightBuf, weightOffset: 0,
                                 convOut: outBuf,
                                 t: UInt32(T), channels: UInt32(C),
                                 kernelSize: UInt32(K))
        kernels.encodeConvWindowUpdate(commandBuffer: cb,
                                       qkv: qkvBuf,
                                       oldWindow: winBuf,
                                       newWindow: stageBuf,
                                       t: UInt32(T), channels: UInt32(C),
                                       kernelSize: UInt32(K))
        if let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: stageBuf, sourceOffset: 0,
                      to: winBuf, destinationOffset: 0,
                      size: windowRows * C * MemoryLayout<Float16>.stride)
            blit.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()

        // CPU reference.
        func inputRow(_ r: Int, c: Int) -> Float {
            r < windowRows ? window[r * C + c] : qkv[(r - windowRows) * C + c]
        }
        for t in 0..<T {
            for c in 0..<C {
                var acc: Float = 0
                for j in 0..<K {
                    acc += Self.bf16ToFloat(weightBits[c * K + j]) * inputRow(t + j, c: c)
                }
                let silu = acc / (1 + exp(-acc))
                #expect(abs(Self.readFP16(outBuf, count: T * C)[t * C + c] - silu) < 2e-2,
                        "conv[\(t),\(c)]")
            }
        }
        for w in 0..<windowRows {
            for c in 0..<C {
                let expected = inputRow(T + w, c: c)
                #expect(abs(Self.readFP16(winBuf, count: windowRows * C)[w * C + c] - expected) < 2e-2,
                        "window[\(w),\(c)]")
            }
        }
    }

    // MARK: - GDN recurrence

    @Test func gdnRecurse_zeroInitialState() throws {
        let T = 1, Hk = 2, Hv = 4, Dk = 128, Dv = 128
        let ctx = try MetalContext()
        let kernels = try LinearAttnKernels(context: ctx)
        var rng = Rng(seed: 0x2024)
        let q = (0..<(T * Hk * Dk)).map { _ in rng.unit() }
        let k = (0..<(T * Hk * Dk)).map { _ in rng.unit() }
        let v = (0..<(T * Hv * Dv)).map { _ in rng.unit() }
        let a = [Float](repeating: 0, count: T * Hv)
        let b = (0..<(T * Hv)).map { _ in rng.unit() * 2 }
        let aLogBits = [UInt16](repeating: 0, count: Hv)
        let dtBits = [UInt16](repeating: 0, count: Hv)
        let state = [Float](repeating: 0, count: Hv * Dv * Dk)
        guard let qBuf = Self.makeFP16(q, device: ctx.device),
              let kBuf = Self.makeFP16(k, device: ctx.device),
              let vBuf = Self.makeFP16(v, device: ctx.device),
              let aBuf = Self.makeFP16(a, device: ctx.device),
              let bBuf = Self.makeFP16(b, device: ctx.device),
              let aLogBuf = ctx.device.makeBuffer(bytes: aLogBits, length: aLogBits.count * 2, options: .storageModeShared),
              let dtBuf = ctx.device.makeBuffer(bytes: dtBits, length: dtBits.count * 2, options: .storageModeShared),
              let stateBuf = ctx.device.makeBuffer(bytes: state, length: state.count * MemoryLayout<Float32>.stride, options: .storageModeShared),
              let yBuf = ctx.device.makeBuffer(length: T * Hv * Dv * MemoryLayout<Float16>.stride, options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernels.encodeGDNRecurse(commandBuffer: cb,
                                 q: qBuf, k: kBuf, v: vBuf,
                                 a: aBuf, b: bBuf,
                                 aLog: aLogBuf, aLogOffset: 0,
                                 dtBias: dtBuf, dtBiasOffset: 0,
                                 state: stateBuf,
                                 y: yBuf,
                                 t: UInt32(T),
                                 numKHeads: UInt32(Hk), numVHeads: UInt32(Hv),
                                 keyHeadDim: UInt32(Dk), valueHeadDim: UInt32(Dv),
                                 scalarStride: UInt32(Hv), qkvStride: UInt32(Hk * Dk), yStride: UInt32(Hv * Dv))
        cb.commit()
        cb.waitUntilCompleted()
        let outState = stateBuf.contents().bindMemory(to: Float32.self, capacity: Hv * Dv * Dk)
        // Reference: zero state -> kvMem = 0, delta = v * beta, s1 = k * v * beta.
        let beta0 = 1 / (1 + exp(-b[0]))
        var worst: Float = 0
        for i in 0..<Dk {
            let expected = Float(Float16(k[i])) * Float(Float16(v[0])) * beta0
            worst = max(worst, abs(outState[i] - expected))
            if i < 4 {
                print("ZS dk=\(i) outState=\(outState[i]) expected=\(expected) "
                    + "k=\(Float(Float16(k[i]))) v0=\(Float(Float16(v[0]))) beta0=\(beta0)")
            }
        }
        for i in 0..<Dk {
            let expected = Float(Float16(k[i])) * Float(Float16(v[0])) * beta0
            #expect(abs(outState[i] - expected) < 1e-2, "zero-state[\(i)]")
        }
        print("ZS y[0]=\(Self.readFP16(yBuf, count: 8)[0])")
        #expect(worst < 1e-2, "zero-state worst=\(worst)")
    }

    /// Debug harness: reruns the GDN kernel on the layer-0 inputs dumped from
    /// a real prefill (`TURBO_QWEN_DUMP_DIR`) and compares against the numpy
    /// reference also dumped there. Skips unless the dump exists.
    @Test func gdnRecurse_realLayer0Inputs() throws {
        let dir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] ?? ""
        let convURL = URL(fileURLWithPath: "\(dir)/l0_conv.f16")
        guard !dir.isEmpty,
              FileManager.default.fileExists(atPath: convURL.path) else { return }
        let T = 13, Hk = 16, Hv = 32, Dk = 128, Dv = 128
        let ctx = try MetalContext()
        let kernels = try LinearAttnKernels(context: ctx)

        func f16raw(_ name: String) throws -> [Float] {
            let data = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name)"))
            var halves = [UInt16](repeating: 0, count: data.count / 2)
            _ = halves.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            return halves.map { Float(Float16(bitPattern: $0)) }
        }
        let conv = try f16raw("l0_conv.f16")
        let ab = try f16raw("l0_ab.f16")
        let aLogRef = try Array(Data(contentsOf: URL(fileURLWithPath: "\(dir)/a_log.f32"))
            .withUnsafeBytes { Array($0.bindMemory(to: Float32.self)) })
        let dtRef = try Array(Data(contentsOf: URL(fileURLWithPath: "\(dir)/dt_bias.f32"))
            .withUnsafeBytes { Array($0.bindMemory(to: Float32.self)) })

        let q = Array(conv[0..<(T * Hk * Dk)])
        let k = Array(conv[(T * Hk * Dk)..<(T * 2 * Hk * Dk)])
        let v = Array(conv[(T * 2 * Hk * Dk)..<(T * (2 * Hk * Dk + Hv * Dv))])
        // The runner's layout: ONE interleaved buffer [T, a(32) | b(32)],
        // b pointer at element 32, token stride 64 elements.
        let a = Array(ab)
        let b = Array(ab)

        let convRow = 2 * Hk * Dk + Hv * Dv
        // One shared conv buffer like the runner: q|k|v contiguous.
        guard let convBuf = Self.makeFP16(conv, device: ctx.device),
              let aBuf = Self.makeFP16(a, device: ctx.device),
              let bBuf = Self.makeFP16(b, device: ctx.device),
              let aLogBuf = ctx.device.makeBuffer(
                bytes: aLogRef, length: aLogRef.count * 4, options: .storageModeShared),
              let dtBuf = ctx.device.makeBuffer(
                bytes: dtRef, length: dtRef.count * 4, options: .storageModeShared),
              let stateBuf = ctx.device.makeBuffer(
                length: Hv * Dv * Dk * 4, options: .storageModeShared),
              let yBuf = ctx.device.makeBuffer(
                length: T * Hv * Dv * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        memset(stateBuf.contents(), 0, stateBuf.length)
        // The kernel expects BF16 A_log/dt_bias; the dump is fp32. Convert.
        let aLogBits = aLogRef.map { Self.bf16Bits($0) }
        let dtBits = dtRef.map { Self.bf16Bits($0) }
        memcpy(aLogBuf.contents(), aLogBits, aLogBits.count * 2)
        memcpy(dtBuf.contents(), dtBits, dtBits.count * 2)

        do {
            var hash: UInt64 = 1469598103934665603
            for f in [conv, a, b] {
                for vv in f {
                    let bits = UInt64(vv.bitPattern)
                    hash = (hash &+ bits) &* 1099511628211
                }
            }
            print("TESTHASH conv/a/b fnv:", hash)
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernels.encodeGDNRecurse(commandBuffer: cb,
                                 q: convBuf, qOffset: 0,
                                 k: convBuf, kOffset: Hk * Dk * 2,
                                 v: convBuf, vOffset: 2 * Hk * Dk * 2,
                                 a: aBuf, b: aBuf, bOffset: Hv * 2,
                                 aLog: aLogBuf, aLogOffset: 0,
                                 dtBias: dtBuf, dtBiasOffset: 0,
                                 state: stateBuf,
                                 y: yBuf,
                                 t: UInt32(T),
                                 numKHeads: UInt32(Hk), numVHeads: UInt32(Hv),
                                 keyHeadDim: UInt32(Dk), valueHeadDim: UInt32(Dv),
                                 scalarStride: UInt32(2 * Hv), qkvStride: UInt32(convRow), yStride: UInt32(Hv * Dv))
        cb.commit()
        cb.waitUntilCompleted()

        let y = Self.readFP16(yBuf, count: T * Hv * Dv)
        let nanCount = y.filter { $0.isNaN }.count
        print("REALINPUT nan count:", nanCount)
        // T-sweep: rerun with growing T to find where the state explodes.
        for sweepT in 1...13 {
            guard let stSweep = ctx.device.makeBuffer(
                    length: Hv * Dv * Dk * 4, options: .storageModeShared),
                  let ySweep = ctx.device.makeBuffer(
                    length: sweepT * Hv * Dv * 2, options: .storageModeShared) else { continue }
            memset(stSweep.contents(), 0, stSweep.length)
            let cbS = ctx.queue.makeCommandBuffer()!
            kernels.encodeGDNRecurse(commandBuffer: cbS,
                                     q: convBuf, qOffset: 0,
                                     k: convBuf, kOffset: Hk * Dk * 2,
                                     v: convBuf, vOffset: 2 * Hk * Dk * 2,
                                     a: aBuf, b: bBuf,
                                     aLog: aLogBuf, aLogOffset: 0,
                                     dtBias: dtBuf, dtBiasOffset: 0,
                                     state: stSweep,
                                     y: ySweep,
                                     t: UInt32(sweepT),
                                     numKHeads: UInt32(Hk), numVHeads: UInt32(Hv),
                                     keyHeadDim: UInt32(Dk), valueHeadDim: UInt32(Dv),
                                     scalarStride: UInt32(Hv), qkvStride: UInt32(Hk * Dk), yStride: UInt32(Hv * Dv))
            cbS.commit()
            cbS.waitUntilCompleted()
            let yRow = Self.readFP16(ySweep, count: sweepT * Hv * Dv)
            let lastY = yRow[(sweepT - 1) * Hv * Dv + 12 * Dv + 79]
            var stMax: Float = 0
            for i in 0..<(Hv * Dv * Dk) {
                stMax = max(stMax, abs(stSweep.contents().load(
                    fromByteOffset: i * 4, as: Float32.self)))
            }
            print("SWEEP T=\(sweepT) y[t-1,12,79]=\(lastY) stateAbsmax=\(stMax)")
            if sweepT == 8, let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                let stData = Data(bytes: stSweep.contents(), count: Hv * Dv * Dk * 4)
                try? stData.write(to: URL(fileURLWithPath:
                    "\(dumpDir)/sweep8_state.f32"))
            }
            if sweepT == 1 {
                // a/b swap probe: pass aBuf as b and bBuf as a.
                guard let stS2 = ctx.device.makeBuffer(
                        length: Hv * Dv * Dk * 4, options: .storageModeShared),
                      let yS2 = ctx.device.makeBuffer(
                        length: Hv * Dv * 2, options: .storageModeShared) else { continue }
                memset(stS2.contents(), 0, stS2.length)
                let cbS2 = ctx.queue.makeCommandBuffer()!
                kernels.encodeGDNRecurse(commandBuffer: cbS2,
                                         q: convBuf, qOffset: 0,
                                         k: convBuf, kOffset: Hk * Dk * 2,
                                         v: convBuf, vOffset: 2 * Hk * Dk * 2,
                                         a: bBuf, b: aBuf,
                                         aLog: aLogBuf, aLogOffset: 0,
                                         dtBias: dtBuf, dtBiasOffset: 0,
                                         state: stS2, y: yS2,
                                         t: 1,
                                         numKHeads: UInt32(Hk), numVHeads: UInt32(Hv),
                                         keyHeadDim: UInt32(Dk), valueHeadDim: UInt32(Dv),
                                         scalarStride: UInt32(Hv),
                                         qkvStride: UInt32(convRow),
                                         yStride: UInt32(Hv * Dv))
                cbS2.commit()
                cbS2.waitUntilCompleted()
                var stMax2: Float = 0
                for i in 0..<(Hv * Dv * Dk) {
                    stMax2 = max(stMax2, abs(stS2.contents().load(
                        fromByteOffset: i * 4, as: Float32.self)))
                }
                print("SWAPPROBE T=1 stateAbsmax=\(stMax2) (numpy a/beta-swap expect 1.898)")
            }
        }
        if let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
            let yHalves: [UInt16] = y.map { Float16($0).bitPattern }
            let yData = yHalves.withUnsafeBufferPointer { ptr in
                Data(bytes: ptr.baseAddress!, count: ptr.count * 2)
            }
            try? yData.write(to: URL(fileURLWithPath:
                "\(dumpDir)/kernelsingle_y.f16"))
            let stData = Data(bytes: stateBuf.contents(), count: Hv * Dv * Dk * 4)
            try? stData.write(to: URL(fileURLWithPath:
                "\(dumpDir)/kernelsingle_state.f32"))
        }
        let refData0 = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/gdn_ref.f32"))
        let ref0 = refData0.withUnsafeBytes { Array($0.bindMemory(to: Float32.self)) }
        var worst0: Float = 0
        var worstIdx = -1
        for i in 0..<(T * Hv * Dv) where !y[i].isNaN {
            let d = abs(y[i] - ref0[i])
            if d.isFinite && d > worst0 { worst0 = d; worstIdx = i }
        }
        print("REALINPUT worst finite diff (first pass):", worst0, "at", worstIdx,
              "y=", worstIdx >= 0 ? y[worstIdx] : 0,
              "ref=", worstIdx >= 0 ? ref0[worstIdx] : 0,
              "t=", worstIdx / (Hv * Dv), "hv=", (worstIdx / Dv) % Hv, "dv=", worstIdx % Dv)
        // Second run on a fresh state buffer: buffer reuse in the runner was
        // once the culprit; this re-run proves the kernel alone is stable.
        guard let stateBuf2 = ctx.device.makeBuffer(
                length: Hv * Dv * Dk * 4, options: .storageModeShared),
              let yBuf2 = ctx.device.makeBuffer(
                length: T * Hv * Dv * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else {
            Issue.record("alloc2 failed"); return
        }
        memset(stateBuf2.contents(), 0, stateBuf2.length)
        let cb2 = ctx.queue.makeCommandBuffer()!
        kernels.encodeGDNRecurse(commandBuffer: cb2,
                                 q: convBuf, qOffset: 0,
                                 k: convBuf, kOffset: Hk * Dk * 2,
                                 v: convBuf, vOffset: 2 * Hk * Dk * 2,
                                 a: aBuf, b: bBuf,
                                 aLog: aLogBuf, aLogOffset: 0,
                                 dtBias: dtBuf, dtBiasOffset: 0,
                                 state: stateBuf2,
                                 y: yBuf2,
                                 t: UInt32(T),
                                 numKHeads: UInt32(Hk), numVHeads: UInt32(Hv),
                                 keyHeadDim: UInt32(Dk), valueHeadDim: UInt32(Dv),
                                 scalarStride: UInt32(Hv), qkvStride: UInt32(Hk * Dk), yStride: UInt32(Hv * Dv))
        cb2.commit()
        cb2.waitUntilCompleted()
        let y2 = Self.readFP16(yBuf2, count: T * Hv * Dv)
        print("REALINPUT second-run nan:", y2.filter { $0.isNaN }.count)
        var worst2: Float = 0
        var worstIdx2 = -1
        var bigCount = 0
        for i in 0..<(T * Hv * Dv) where !y2[i].isNaN {
            let d = abs(y2[i] - ref0[i])
            if d > 0.5 { bigCount += 1 }
            if d.isFinite && d > worst2 { worst2 = d; worstIdx2 = i }
        }
        print("REALINPUT second-run worst finite diff:", worst2, "at", worstIdx2,
              "t=", worstIdx2 / (Hv * Dv), "hv=", (worstIdx2 / Dv) % Hv,
              "dv=", worstIdx2 % Dv, "y=", worstIdx2 >= 0 ? y2[worstIdx2] : 0,
              "ref=", worstIdx2 >= 0 ? ref0[worstIdx2] : 0, "bigCount:", bigCount)
        // Also dump what the kernel read for token 0 hv=0 dv=0 via a micro
        // single-token dispatch (t=1 on the first token's inputs only).
        if let q1 = ctx.device.makeBuffer(length: 128 * 2, options: .storageModeShared),
           let k1 = ctx.device.makeBuffer(length: 128 * 2, options: .storageModeShared),
           let v1 = ctx.device.makeBuffer(length: 128 * 2, options: .storageModeShared),
           let a1 = ctx.device.makeBuffer(length: 32 * 2, options: .storageModeShared),
           let b1 = ctx.device.makeBuffer(length: 32 * 2, options: .storageModeShared),
           let st1 = ctx.device.makeBuffer(length: Hv * Dv * Dk * 4, options: .storageModeShared),
           let y1 = ctx.device.makeBuffer(length: Dv * 2, options: .storageModeShared) {
            memset(st1.contents(), 0, st1.length)
            memcpy(q1.contents(), convBuf.contents(), 256)
            memcpy(k1.contents(), convBuf.contents().advanced(by: 4096), 256)
            memcpy(v1.contents(), convBuf.contents().advanced(by: 8192), 256)
            memcpy(a1.contents(), aBuf.contents(), 32 * 2)
            memcpy(b1.contents(), bBuf.contents(), 32 * 2)
            var tokenStates: [Float] = []
            var tokenY: [Float] = []
            for tok in 0..<T {
                memcpy(q1.contents(),
                       convBuf.contents().advanced(by: tok * convRow * 2), 256)
                memcpy(k1.contents(),
                       convBuf.contents().advanced(by: tok * convRow * 2 + 4096), 256)
                memcpy(v1.contents(),
                       convBuf.contents().advanced(by: tok * convRow * 2 + 8192), 256)
                memcpy(a1.contents(),
                       aBuf.contents().advanced(by: tok * Hv * 2), 32 * 2)
                memcpy(b1.contents(),
                       bBuf.contents().advanced(by: tok * Hv * 2), 32 * 2)
                let cb3 = ctx.queue.makeCommandBuffer()!
                kernels.encodeGDNRecurse(commandBuffer: cb3,
                                         q: q1, k: k1, v: v1,
                                         a: a1, b: b1,
                                         aLog: aLogBuf, aLogOffset: 0,
                                         dtBias: dtBuf, dtBiasOffset: 0,
                                         state: st1, y: y1,
                                         t: 1,
                                         numKHeads: UInt32(Hk), numVHeads: UInt32(Hv),
                                         keyHeadDim: UInt32(Dk), valueHeadDim: UInt32(Dv),
                                         scalarStride: UInt32(Hv), qkvStride: UInt32(Hk * Dk), yStride: UInt32(Hv * Dv))
                cb3.commit()
                cb3.waitUntilCompleted()
                let yv = Self.readFP16(y1, count: Dv)
                var mxState: Float = 0
                for i in 0..<(Hv * Dv * Dk) {
                    mxState = max(mxState, abs(st1.contents().load(
                        fromByteOffset: i * 4, as: Float32.self)))
                }
                tokenStates.append(mxState)
                tokenY.append(yv[0])
                if let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] {
                    let stData = Data(bytes: st1.contents(), count: Hv * Dv * Dk * 4)
                    try? stData.write(to: URL(fileURLWithPath:
                        "\(dumpDir)/micro_state_t\(tok).f32"))
                }
                if tok == 0 {
                    // Scan for the largest |state| element and report (hv,dv,dk).
                    var bestF: Float = 0
                    var bestIdx = 0
                    for i in 0..<(Hv * Dv * Dk) {
                        let v = st1.contents().load(fromByteOffset: i * 4,
                                                    as: Float32.self)
                        if abs(v) > bestF { bestF = abs(v); bestIdx = i }
                    }
                    print("KERNEL t0 state absmax:", bestF, "at hv=",
                          bestIdx / (Dv * Dk), "dv=", (bestIdx / Dk) % Dv,
                          "dk=", bestIdx % Dk)
                    let dumpDir0 = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"]!
                    let stData = Data(bytes: st1.contents(), count: Hv * Dv * Dk * 4)
                    try? stData.write(to: URL(fileURLWithPath:
                        "\(dumpDir0)/micro_state_t0.f32"))
                }
            }
            let dumpDir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"]!
            // Kernel per-token y for hv=0, dv=0..127:
            var allY: [Float] = []
            for tok in 0..<T {
                // rerun token tok with state snapshot? Simpler: save the
                // per-token y rows we saw: rerun the loop writing y rows.
                _ = tok
            }
            // Save inputs as the kernel saw them (the conv and ab buffers).
            let convData = Data(bytes: convBuf.contents(),
                                count: T * convRow * 2)
            try? convData.write(to: URL(fileURLWithPath: "\(dumpDir)/kernel_conv.f16"))
            let abData = Data(bytes: aBuf.contents(), count: T * Hv * 2)
            try? abData.write(to: URL(fileURLWithPath: "\(dumpDir)/kernel_a.f16"))
            let abBData = Data(bytes: bBuf.contents(), count: T * Hv * 2)
            try? abBData.write(to: URL(fileURLWithPath: "\(dumpDir)/kernel_b.f16"))
            print("PER-TOKEN y[0]:", tokenY.map { String(format: "%.5g", $0) }.joined(separator: " "))
            print("PER-TOKEN state absmax:", tokenStates.map { String(format: "%.4g", $0) }.joined(separator: " "))
            let yv = Self.readFP16(y1, count: Dv)
            let base = st1.contents()
            let s0 = base.load(as: Float32.self)
            let s1 = base.load(fromByteOffset: 4, as: Float32.self)
            let s2 = base.load(fromByteOffset: 8, as: Float32.self)
            let s3 = base.load(fromByteOffset: 12, as: Float32.self)
            func u16(_ buf: MTLBuffer, _ i: Int) -> UInt16 {
                buf.contents().load(fromByteOffset: i * 2, as: UInt16.self)
            }
            func f16bits(_ buf: MTLBuffer, _ i: Int) -> Float {
                Float(Float16(bitPattern: u16(buf, i)))
            }
            let qv0 = f16bits(q1, 0)
            let kv0 = f16bits(k1, 0)
            let vv0 = f16bits(v1, 0)
            // Cross-check against the CURRENT conv file the numpy ref uses.
            let convNow = try f16raw("l0_conv.f16")
            print("MICRO file q0:", convNow[0], "k0:", convNow[2048],
                  "v0:", convNow[4096])
            let av0 = f16bits(a1, 0)
            let bv0 = f16bits(b1, 0)
            let alv0 = Quantization.bf16ToFloat(u16(aLogBuf, 0))
            let dtv0 = Quantization.bf16ToFloat(u16(dtBuf, 0))
            print("MICRO q0:", qv0, "k0:", kv0, "v0:", vv0,
                  "a0:", av0, "b0:", bv0, "alog0:", alv0, "dt0:", dtv0)
            var bvals: [Float] = []
            for i in 0..<8 { bvals.append(f16bits(b1, i)) }
            print("MICRO b1[0..8]:", bvals)
            print("MICRO y[0,0,:4]:", yv[0], yv[1], yv[2], yv[3],
                  "state[0,0,:4]:", s0, s1, s2, s3)
        }
        let refData = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/gdn_ref.f32"))
        let ref = refData.withUnsafeBytes { Array($0.bindMemory(to: Float32.self)) }
        var worst: Float = 0
        for i in 0..<(T * Hv * Dv) where !y[i].isNaN {
            worst = max(worst, abs(y[i] - ref[i]))
        }
        print("REALINPUT worst finite diff:", worst)
        #expect(nanCount == 0, "nan count \(nanCount)")
    }

    @Test func attentionDecodeGenericGeometry() throws {
        // The generic (non-specialized) decode path with the Qwen full-layer
        // geometry: NQ=16, NKV=2, headDim=256.
        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        let NQ = 16, NKV = 2, HD = 256, S = 3
        var rng = Rng(seed: 0x4041)
        let q = (0..<(NQ*HD)).map { _ in rng.unit() }
        let k = (0..<(S*NKV*HD)).map { _ in rng.unit() }
        let v = (0..<(S*NKV*HD)).map { _ in rng.unit() }
        guard let qBuf = Self.makeFP16(q, device: ctx.device),
              let kBuf = Self.makeFP16(k, device: ctx.device),
              let vBuf = Self.makeFP16(v, device: ctx.device),
              let oBuf = ctx.device.makeBuffer(length: NQ*HD*2,
                                               options: .storageModeShared) else {
            Issue.record("alloc"); return
        }
        memset(oBuf.contents(), 0, oBuf.length)
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeFull(commandBuffer: cb,
                          q: qBuf, k: kBuf, v: vBuf, out: oBuf,
                          headDim: UInt32(HD), numQHeads: UInt32(NQ),
                          numKVHeads: UInt32(NKV), seqLen: UInt32(S),
                          scale: 1.0/16.0)
        cb.commit()
        cb.waitUntilCompleted()
        var out = [Float](repeating: 0, count: NQ*HD)
        for i in 0..<(NQ*HD) {
            out[i] = Float(oBuf.contents().load(fromByteOffset: i*2, as: Float16.self))
        }
        var worst: Float = 0
        for h in 0..<NQ {
            let kvh = h/(NQ/NKV)
            var scores: [Double] = []
            for p in 0..<S {
                var dot = 0.0
                for d in 0..<HD { dot += Double(q[h*HD+d]) * Double(k[p*NKV*HD + kvh*HD + d]) }
                scores.append(dot/16.0)
            }
            let m = scores.max()!
            let ex = scores.map { exp($0 - m) }
            let sum = ex.reduce(0, +)
            for d in 0..<HD {
                var acc = 0.0
                for p in 0..<S { acc += ex[p]/sum * Double(v[p*NKV*HD + kvh*HD + d]) }
                worst = max(worst, abs(out[h*HD+d] - Float(acc)))
            }
        }
        #expect(worst < 5e-3, "generic decode attention worst=\(worst)")
    }

    @Test func gdnRecurse_matchesCPUReference() throws {
        let T = 1, Hk = 2, Hv = 4, Dk = 128, Dv = 128
        let ctx = try MetalContext()
        let kernels = try LinearAttnKernels(context: ctx)
        var rng = Rng(seed: 0x9D07)

        let q = (0..<(T * Hk * Dk)).map { _ in rng.unit() }
        let k = (0..<(T * Hk * Dk)).map { _ in rng.unit() }
        let v = (0..<(T * Hv * Dv)).map { _ in rng.unit() }
        let a = (0..<(T * Hv)).map { _ in rng.unit() * 2 }
        let b = (0..<(T * Hv)).map { _ in rng.unit() * 2 }
        let aLogBits = (0..<Hv).map { _ in rng.bf16() }
        let dtBits = (0..<Hv).map { _ in rng.bf16() }
        var state = (0..<(Hv * Dv * Dk)).map { _ in rng.unit() * 0.25 }

        guard let qBuf = Self.makeFP16(q, device: ctx.device),
              let kBuf = Self.makeFP16(k, device: ctx.device),
              let vBuf = Self.makeFP16(v, device: ctx.device),
              let aBuf = Self.makeFP16(a, device: ctx.device),
              let bBuf = Self.makeFP16(b, device: ctx.device),
              let aLogBuf = ctx.device.makeBuffer(
                bytes: aLogBits, length: aLogBits.count * 2,
                options: .storageModeShared),
              let dtBuf = ctx.device.makeBuffer(
                bytes: dtBits, length: dtBits.count * 2,
                options: .storageModeShared),
              let stateBuf = ctx.device.makeBuffer(
                bytes: state, length: state.count * MemoryLayout<Float32>.stride,
                options: .storageModeShared),
              let yBuf = ctx.device.makeBuffer(
                length: T * Hv * Dv * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernels.encodeGDNRecurse(commandBuffer: cb,
                                 q: qBuf, k: kBuf, v: vBuf,
                                 a: aBuf, b: bBuf,
                                 aLog: aLogBuf, aLogOffset: 0,
                                 dtBias: dtBuf, dtBiasOffset: 0,
                                 state: stateBuf,
                                 y: yBuf,
                                 t: UInt32(T),
                                 numKHeads: UInt32(Hk), numVHeads: UInt32(Hv),
                                 keyHeadDim: UInt32(Dk), valueHeadDim: UInt32(Dv),
                                 scalarStride: UInt32(Hv), qkvStride: UInt32(Hk * Dk), yStride: UInt32(Hv * Dv))
        cb.commit()
        cb.waitUntilCompleted()

        // CPU reference (gated_delta_ops): fp32 state, per-v-head scalar g.
        var s = state
        for t in 0..<T {
            for hv in 0..<Hv {
                let hk = hv / (Hv / Hk)
                let g = exp(-exp(Self.bf16ToFloat(aLogBits[hv]))
                            * Self.softplus(Float(a[t * Hv + hv])
                                       + Self.bf16ToFloat(dtBits[hv])))
                let beta = 1 / (1 + exp(-b[t * Hv + hv]))
                let kt = Array(k[(t * Hk + hk) * Dk..<(t * Hk + hk + 1) * Dk])
                    .map { Float(Float16($0)) }
                let qt = Array(q[(t * Hk + hk) * Dk..<(t * Hk + hk + 1) * Dk])
                    .map { Float(Float16($0)) }
                for dv in 0..<Dv {
                    let base = (hv * Dv + dv) * Dk
                    for i in 0..<Dk { s[base + i] *= g }
                    var kvMem: Float = 0
                    for i in 0..<Dk { kvMem += s[base + i] * kt[i] }
                    let delta = (Float(Float16(v[t * Hv * Dv + hv * Dv + dv])) - kvMem) * beta
                    for i in 0..<Dk { s[base + i] += kt[i] * delta }
                    var out: Float = 0
                    for i in 0..<Dk { out += s[base + i] * qt[i] }
                    #expect(abs(Self.readFP16(yBuf, count: T * Hv * Dv)[t * Hv * Dv + hv * Dv + dv] - out)
                            < 2e-2 * max(1, abs(out)),
                            "y[\(t),\(hv),\(dv)]")
                }
            }
        }
        // State check on a spread of rows.
        let outState = stateBuf.contents().bindMemory(to: Float32.self,
                                                      capacity: Hv * Dv * Dk)
        for probe in 0..<(Hv * Dv * Dk) {
            #expect(abs(outState[probe] - s[probe]) < 1e-4, "state[\(probe)]")
        }
    }

    @Test func gdnGatedNorm_matchesCPUReference() throws {
        let T = 3, Hv = 4, Dv = 128
        let ctx = try MetalContext()
        let kernels = try LinearAttnKernels(context: ctx)
        var rng = Rng(seed: 0x61A7)
        let y = (0..<(T * Hv * Dv)).map { _ in rng.unit() }
        let z = (0..<(T * Hv * Dv)).map { _ in rng.unit() * 2 }
        let wBits = (0..<Dv).map { _ in rng.bf16() }
        guard let yBuf = Self.makeFP16(y, device: ctx.device),
              let zBuf = Self.makeFP16(z, device: ctx.device),
              let wBuf = ctx.device.makeBuffer(
                bytes: wBits, length: wBits.count * 2, options: .storageModeShared),
              let outBuf = ctx.device.makeBuffer(
                length: T * Hv * Dv * MemoryLayout<Float16>.stride,
                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernels.encodeGDNGatedNorm(commandBuffer: cb,
                                   y: yBuf, z: zBuf,
                                   weight: wBuf, weightOffset: 0,
                                   out: outBuf,
                                   t: UInt32(T), numVHeads: UInt32(Hv),
                                   valueHeadDim: UInt32(Dv),
                                   eps: 1e-6)
        cb.commit()
        cb.waitUntilCompleted()
        let out = Self.readFP16(outBuf, count: T * Hv * Dv)
        for t in 0..<T {
            for h in 0..<Hv {
                let base = (t * Hv + h) * Dv
                let yh = (0..<Dv).map { Float(Float16(y[base + $0])) }
                let mean = yh.reduce(0, +) / Float(Dv)
                let inv = 1 / (Self.meanSquare(yh, mean: mean) + 1e-6).squareRoot()
                for i in 0..<Dv {
                    let gate = {
                        let g = z[base + i]
                        return g / (1 + exp(-g))
                    }()
                    let expected = gate * yh[i] * inv * Self.bf16ToFloat(wBits[i])
                    #expect(abs(out[base + i] - expected) < 3e-2,
                            "gated[\(t),\(h),\(i)]")
                }
            }
        }
    }

    private static func meanSquare(_ v: [Float], mean: Float) -> Float {
        // rms_norm semantics: mean of squares (the `mean` argument is unused
        // by the reference; kept for call-site symmetry).
        _ = mean
        return v.reduce(0) { $0 + $1 * $1 } / Float(v.count)
    }

    private static func softplus(_ x: Float) -> Float {
        x > 20 ? x : log(1 + exp(x))
    }

    // MARK: - Full-attention epilogue (norm + partial RoPE + sigmoid gate)

    @Test func attnEpilogue_matchesCPUReference() throws {
        let T = 2, NQ = 4, NKV = 1, HD = 64, RP = 16 // rotaryDim 32
        let ctx = try MetalContext()
        let kernels = try LinearAttnKernels(context: ctx)
        var rng = Rng(seed: 0xE911)
        let qRows = NQ * HD * 2
        let theta: Float = 1_000_000
        let q = (0..<(T * qRows)).map { _ in rng.unit() }
        let k = (0..<(T * NKV * HD)).map { _ in rng.unit() }
        let qwBits = (0..<HD).map { _ in rng.bf16() }
        let kwBits = (0..<HD).map { _ in rng.bf16() }
        guard let qBuf = Self.makeFP16(q, device: ctx.device),
              let qOutBuf = ctx.device.makeBuffer(
                length: T * NQ * HD * MemoryLayout<Float16>.stride,
                options: .storageModeShared),
              let kBuf = Self.makeFP16(k, device: ctx.device),
              let qwBuf = ctx.device.makeBuffer(bytes: qwBits, length: qwBits.count * 2,
                                                options: .storageModeShared),
              let kwBuf = ctx.device.makeBuffer(bytes: kwBits, length: kwBits.count * 2,
                                                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernels.encodeAttnEpilogue(commandBuffer: cb,
                                   q: qBuf, qOffset: 0,
                                   qOut: qOutBuf,
                                   qInStride: UInt32(qRows),
                                   qOutStride: UInt32(NQ * HD),
                                   k: kBuf, kOffset: 0,
                                   qWeight: qwBuf, qWeightOffset: 0,
                                   kWeight: kwBuf, kWeightOffset: 0,
                                   startPosition: 3,
                                   t: UInt32(T),
                                   headDim: UInt32(HD),
                                   numQHeads: UInt32(NQ),
                                   numKVHeads: UInt32(NKV),
                                   rotatedPairs: UInt32(RP),
                                   theta: theta,
                                   eps: 1e-6)
        cb.commit()
        cb.waitUntilCompleted()

        let qOut = Self.readFP16(qOutBuf, count: T * NQ * HD)
        let kOut = Self.readFP16(kBuf, count: T * NKV * HD)
        for t in 0..<T {
            for h in 0..<NQ {
                let inBase = t * qRows + h * HD * 2
                let outBase = t * NQ * HD + h * HD
                var normed = (0..<HD).map { Float(Float16(q[inBase + $0])) }
                let mean = normed.reduce(0, +) / Float(HD)
                let inv = 1 / (Self.meanSquare(normed, mean: mean) + 1e-6).squareRoot()
                normed = normed.enumerated().map { Float(Float16($1 * inv * Self.bf16ToFloat(qwBits[$0]))) }
                for p in 0..<RP {
                    let angle = Float(3 + t)
                        * pow(theta, -Float(2 * p) / Float(RP * 2))
                    let c = cos(angle), s2 = sin(angle)
                    let x0 = normed[p], x1 = normed[RP + p]
                    normed[p] = x0 * c - x1 * s2
                    normed[RP + p] = x0 * s2 + x1 * c
                }
                for i in 0..<HD {
                    #expect(abs(qOut[outBase + i] - normed[i]) < 3e-2,
                            "q[\(t),\(h),\(i)]")
                }
            }
            for kh in 0..<NKV {
                let base = t * NKV * HD + kh * HD
                var normed = (0..<HD).map { Float(Float16(k[base + $0])) }
                let mean = normed.reduce(0, +) / Float(HD)
                let inv = 1 / (Self.meanSquare(normed, mean: mean) + 1e-6).squareRoot()
                normed = normed.enumerated().map { Float(Float16($1 * inv * Self.bf16ToFloat(kwBits[$0]))) }
                for p in 0..<RP {
                    let angle = Float(3 + t)
                        * pow(theta, -Float(2 * p) / Float(RP * 2))
                    let c = cos(angle), s2 = sin(angle)
                    let x0 = normed[p], x1 = normed[RP + p]
                    normed[p] = x0 * c - x1 * s2
                    normed[RP + p] = x0 * s2 + x1 * c
                }
                for i in 0..<HD {
                    #expect(abs(kOut[base + i] - normed[i]) < 8e-2,
                            "k[\(t),\(kh),\(i)]")
                }
            }
        }
    }
}
extension LinearAttnKernelTests {
    /// Debug: run the production attention-decode kernel on the layer-3
    /// dumps (`TURBO_QWEN_DUMP_DIR`) and compare against a NumPy reference.
    @Test func attentionRealLayer3Inputs() throws {
        let dir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] ?? ""
        guard !dir.isEmpty,
              FileManager.default.fileExists(atPath: "\(dir)/l3_q.f16") else { return }
        let ctx = try MetalContext()
        let kernel = try Attention(context: ctx)
        let T = 18, NQ = 16, NKV = 2, HD = 256
        func f16raw(_ name: String) throws -> [Float] {
            let data = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name)"))
            var halves = [UInt16](repeating: 0, count: data.count / 2)
            _ = halves.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            return halves.map { Float(Float16(bitPattern: $0)) }
        }
        let q = try f16raw("l3_q.f16")
        let ck = try f16raw("l3_cacheK.f16")
        let cv = try f16raw("l3_cacheV.f16")
        guard let qBuf = Self.makeFP16(q, device: ctx.device),
              let kBuf = Self.makeFP16(ck, device: ctx.device),
              let vBuf = Self.makeFP16(cv, device: ctx.device),
              let oBuf = ctx.device.makeBuffer(length: T*NQ*HD*2,
                                               options: .storageModeShared) else {
            Issue.record("alloc"); return
        }
        // Per token: decode-style single-token attention over [0, t].
        for t in [0, 1, 5] {
            let qRow = t*NQ*HD
            let oRow = t*NQ*HD
            memset(oBuf.contents(), 0, oBuf.length)
            let cb = ctx.queue.makeCommandBuffer()!
            kernel.encodeFull(commandBuffer: cb,
                              q: qBuf, qOffset: qRow*2,
                              k: kBuf, kOffset: 0,
                              v: vBuf, vOffset: 0,
                              out: oBuf, outOffset: oRow*2,
                              headDim: UInt32(HD),
                              numQHeads: UInt32(NQ),
                              numKVHeads: UInt32(NKV),
                              seqLen: UInt32(t+1),
                              scale: 1.0/16.0)
            cb.commit()
            cb.waitUntilCompleted()
            var out = [Float](repeating: 0, count: NQ*HD)
            let base = oBuf.contents().advanced(by: oRow*2)
            for i in 0..<(NQ*HD) {
                out[i] = Float(base.load(fromByteOffset: i*2, as: Float16.self))
            }
            // numpy
            var ref = Array(repeating: Float(0), count: NQ*HD)
            for h in 0..<NQ {
                let kvh = h/8
                var scores: [Double] = []
                for p in 0...(t) {
                    var dot = 0.0
                    for d in 0..<HD { dot += Double(q[qRow + h*HD + d]) * Double(ck[p*NKV*HD + kvh*HD + d]) }
                    scores.append(dot/16.0)
                }
                let m = scores.max()!
                let ex = scores.map { exp($0 - m) }
                let sum = ex.reduce(0, +)
                for d in 0..<HD {
                    var acc = 0.0
                    for p in 0...(t) { acc += ex[p]/sum * Double(cv[p*NKV*HD + kvh*HD + d]) }
                    ref[h*HD + d] = Float(acc)
                }
            }
            var worst: Float = 0
            var nz = 0
            for i in 0..<(NQ*HD) {
                if out[i] != 0 { nz += 1 }
                worst = max(worst, abs(out[i] - ref[i]))
            }
            // Pass-1 diagnostics: read the split-KV scratch.
            func f32at(_ buf: MTLBuffer, _ i: Int) -> Float {
                buf.contents().load(fromByteOffset: i * 4, as: Float32.self)
            }
            var oScratch: [Float] = []
            for i in 0..<4 {
                oScratch.append(kernel.oPartialDebug.contents().load(
                    fromByteOffset: i*4, as: Float32.self))
            }
            print("  oPartial[0..4]:", oScratch)
            print("  mPartial[0..4]:", f32at(kernel.mPartialDebug, 0),
                  f32at(kernel.mPartialDebug, 1),
                  f32at(kernel.mPartialDebug, 2),
                  f32at(kernel.mPartialDebug, 3),
                  " dPartial:", f32at(kernel.dPartialDebug, 0),
                  f32at(kernel.dPartialDebug, 1),
                  f32at(kernel.dPartialDebug, 2),
                  f32at(kernel.dPartialDebug, 3))
            print("ATTN t=\(t): swift-vs-numpy worst:", worst, "nonzero:", nz, "/", NQ*HD)
            print("  swift[:6]:", out[0..<6].map { String(format: "%.4g", $0) }.joined(separator: " "))
            print("  ref  [:6]:", ref[0..<6].map { String(format: "%.4g", $0) }.joined(separator: " "))
            print("  qBuf[:4]:", q[qRow..<(qRow+4)].map { String(format: "%.4g", $0) }.joined(separator: " "),
                  "ck[:4]:", ck[0..<4].map { String(format: "%.4g", $0) }.joined(separator: " "))
        }
    }
}

extension LinearAttnKernelTests {
    /// Debug: the PREFILL causal-tiled attention kernel on the layer-3
    /// dumps vs NumPy, for the Qwen full-layer geometry (16Q/2KV, HD 256).
    @Test func attentionPrefillRealLayer3() throws {
        let dir = ProcessInfo.processInfo.environment["TURBO_QWEN_DUMP_DIR"] ?? ""
        guard !dir.isEmpty,
              FileManager.default.fileExists(atPath: "\(dir)/l3_q.f16") else { return }
        let ctx = try MetalContext()
        let kernel = try PrefillAttention(context: ctx)
        let T = 18, NQ = 16, NKV = 2, HD = 256
        func f16raw(_ name: String) throws -> [Float] {
            let data = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name)"))
            var halves = [UInt16](repeating: 0, count: data.count / 2)
            _ = halves.withUnsafeMutableBytes { data.copyBytes(to: $0) }
            return halves.map { Float(Float16(bitPattern: $0)) }
        }
        let q = try f16raw("l3_q.f16")
        let ck = try f16raw("l3_cacheK.f16")
        let cv = try f16raw("l3_cacheV.f16")
        guard let qBuf = Self.makeFP16(q, device: ctx.device),
              let kBuf = Self.makeFP16(ck, device: ctx.device),
              let vBuf = Self.makeFP16(cv, device: ctx.device),
              let oBuf = ctx.device.makeBuffer(length: T*NQ*HD*2,
                                               options: .storageModeShared) else {
            Issue.record("alloc"); return
        }
        memset(oBuf.contents(), 0, oBuf.length)
        let params = PrefillAttentionParams(
            startPosition: 0,
            queryCount: UInt32(T),
            headDim: UInt32(HD),
            numQHeads: UInt32(NQ),
            numKVHeads: UInt32(NKV),
            kvValidCount: UInt32(T),
            slidingWindow: UInt32(T),
            kvTokenStrideElements: UInt32(NKV*HD),
            qTokenStrideElements: UInt32(NQ*HD),
            oTokenStrideElements: UInt32(NQ*HD),
            scale: 1.0/16.0,
            bidirectionalBlockStart: 0,
            bidirectionalBlockEnd: 0)
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeCausal(commandBuffer: cb,
                            q: qBuf, k: kBuf, v: vBuf, out: oBuf,
                            params: params,
                            kvRingCapacity: 0,
                            layerKind: .full,
                            path: .causalTiled)
        cb.commit()
        cb.waitUntilCompleted()
        var out = [Float](repeating: 0, count: T*NQ*HD)
        for i in 0..<(T*NQ*HD) {
            out[i] = Float(oBuf.contents().load(fromByteOffset: i*2, as: Float16.self))
        }
        // numpy ref
        var worst: Float = 0
        var worstAt = ""
        for t in 0..<T {
            for h in 0..<NQ {
                let kvh = h/(NQ/NKV)
                var scores: [Double] = []
                for p in 0...t {
                    var dot = 0.0
                    for d in 0..<HD {
                        dot += Double(q[t*NQ*HD + h*HD + d]) * Double(ck[p*NKV*HD + kvh*HD + d])
                    }
                    scores.append(dot/16.0)
                }
                let m = scores.max()!
                let ex = scores.map { exp($0 - m) }
                let sum = ex.reduce(0, +)
                for d in 0..<HD {
                    var acc = 0.0
                    for p in 0...t {
                        acc += ex[p]/sum * Double(cv[p*NKV*HD + kvh*HD + d])
                    }
                    let diff = abs(out[t*NQ*HD + h*HD + d] - Float(acc))
                    if diff > worst {
                        worst = diff
                        worstAt = "t=\(t) h=\(h) d=\(d)"
                    }
                }
            }
        }
        print("PREFILL-ATTN worst:", worst, worstAt)
        #expect(worst < 5e-3, "prefill attention worst=\(worst)")
    }
}
