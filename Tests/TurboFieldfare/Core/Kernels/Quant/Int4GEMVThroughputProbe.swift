import Testing
import Foundation
import Metal
@testable import TurboFieldfare

/// Experiment probe, skipped unless TURBO_PERF_PROBE is set. Times the int4
/// GEMV at Qwen3.6 decode shapes, back-to-back versus one kernel per
/// committed-and-waited command buffer, and prints effective GB/s.
@Suite struct Int4GEMVThroughputProbe {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBO_PERF_PROBE"] != nil))
    func probe() throws {
        let ctx = try MetalContext()
        let kernel = try DequantInt4GEMV(context: ctx)
        for (m, n) in [(8192, 2048), (4096, 2048), (2048, 4096), (512, 2048)] {
            let copies = 6   // rotate distinct matrices so the SLC cannot hold them
            let wBytes = m * n / 2, sBytes = m * (n / 64) * 2
            let w = (0..<copies).map { _ in ctx.device.makeBuffer(length: wBytes, options: .storageModeShared)! }
            let s = (0..<copies).map { _ in ctx.device.makeBuffer(length: sBytes, options: .storageModeShared)! }
            let b = (0..<copies).map { _ in ctx.device.makeBuffer(length: sBytes, options: .storageModeShared)! }
            for buf in w + s + b { memset(buf.contents(), 0x11, buf.length) }
            let x = ctx.device.makeBuffer(length: n * 2, options: .storageModeShared)!
            let y = ctx.device.makeBuffer(length: m * 2, options: .storageModeShared)!
            let calls = 240
            func encode(_ cb: MTLCommandBuffer, _ i: Int) {
                kernel.encode(commandBuffer: cb, weights: w[i % copies], scales: s[i % copies],
                              biases: b[i % copies], x: x, y: y, m: UInt32(m), n: UInt32(n))
            }
            // Warm up.
            let warm = ctx.queue.makeCommandBuffer()!
            for i in 0..<calls { encode(warm, i) }
            warm.commit(); warm.waitUntilCompleted()

            let batched = ctx.queue.makeCommandBuffer()!
            for i in 0..<calls { encode(batched, i) }
            batched.commit(); batched.waitUntilCompleted()
            let batchedPer = (batched.gpuEndTime - batched.gpuStartTime) / Double(calls)

            var burstGPU = 0.0
            let wallStart = Date()
            for i in 0..<calls {
                let cb = ctx.queue.makeCommandBuffer()!
                encode(cb, i)
                cb.commit(); cb.waitUntilCompleted()
                burstGPU += cb.gpuEndTime - cb.gpuStartTime
            }
            let burstWall = Date().timeIntervalSince(wallStart) / Double(calls)
            let burstPer = burstGPU / Double(calls)
            let bytes = Double(wBytes + 2 * sBytes)
            print(String(format: "[probe int4 m=%d n=%d] batched %.1f us (%.0f GB/s) | one-per-CB gpu %.1f us (%.0f GB/s), wall %.1f us",
                         m, n, batchedPer * 1e6, bytes / batchedPer / 1e9,
                         burstPer * 1e6, bytes / burstPer / 1e9, burstWall * 1e6))
        }
    }

    /// Fixed GPU cost of opening one compute encoder per tiny dispatch versus
    /// sharing one encoder, for a dependent chain of small kernels.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["TURBO_PERF_PROBE"] != nil))
    func encoderOverhead() throws {
        let ctx = try MetalContext()
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void bump(device float* v [[buffer(0)]], uint i [[thread_position_in_grid]]) { v[i] += 1.0f; }
        """
        let lib = try ctx.device.makeLibrary(source: src, options: nil)
        let pso = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "bump")!)
        let v = ctx.device.makeBuffer(length: 2048 * 4, options: .storageModeShared)!
        let n = 800
        func run(_ shared: Bool) -> Double {
            let cb = ctx.queue.makeCommandBuffer()!
            var enc: MTLComputeCommandEncoder? = shared ? cb.makeComputeCommandEncoder() : nil
            for _ in 0..<n {
                let e = enc ?? cb.makeComputeCommandEncoder()!
                e.setComputePipelineState(pso)
                e.setBuffer(v, offset: 0, index: 0)
                e.dispatchThreads(MTLSize(width: 2048, height: 1, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                if !shared { e.endEncoding() }
            }
            enc?.endEncoding(); enc = nil
            cb.commit(); cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) / Double(n)
        }
        _ = run(true); _ = run(false)
        let one = run(true), many = run(false)
        print(String(format: "[probe encoders] per dispatch: shared encoder %.2f us, encoder-per-dispatch %.2f us",
                     one * 1e6, many * 1e6))
    }
}
