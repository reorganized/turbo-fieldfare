import Synchronization
import Testing
@testable import TurboFieldfareServerCore

private actor TestGate {
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        await withCheckedContinuation { continuation = $0 }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

@Suite("Server coordinator")
struct ServerCoordinatorTests {
    @Test func boundsFIFOAndRecoversAfterCancellation() async throws {
        let coordinator = ServerCoordinator(queueLimit: 1)
        let gate = TestGate()
        let active = Task {
            try await coordinator.run {
                await gate.wait()
                return 1
            }
        }
        while await !coordinator.isActive {
            await Task.yield()
        }
        let queued = Task {
            try await coordinator.run { 2 }
        }
        while await coordinator.queuedCount != 1 {
            await Task.yield()
        }
        await #expect(throws: ServerRequestError.queueFull) {
            try await coordinator.run { 3 }
        }
        queued.cancel()
        _ = try? await queued.value
        await gate.open()
        #expect(try await active.value == 1)
        #expect(await coordinator.queuedCount == 0)
    }

    @Test func activeRunBlocksModelSwitch() async throws {
        let coordinator = ServerCoordinator(queueLimit: 4, currentModelID: "model-a")
        let gate = TestGate()
        let active = Task {
            try await coordinator.run {
                await gate.wait()
                return 1
            }
        }
        while await !coordinator.isActive {
            await Task.yield()
        }
        await #expect(throws: ServerRequestError.modelBusy) {
            try await coordinator.runPreparing(
                requestedModel: "model-b",
                activate: { _ in Issue.record("must not activate while busy") },
                prepare: { () },
                operation: { _ in 2 })
        }
        await gate.open()
        #expect(try await active.value == 1)
        #expect(await coordinator.currentModelID == "model-a")
    }

    @Test func idleRunSwitchesModel() async throws {
        let coordinator = ServerCoordinator(queueLimit: 4, currentModelID: "model-a")
        let activated = Mutex<[String]>([])
        _ = try await coordinator.runPreparing(
            requestedModel: "model-b",
            activate: { id in activated.withLock { $0.append(id) } },
            prepare: { () },
            operation: { _ in 2 })
        #expect(await coordinator.currentModelID == "model-b")
        #expect(activated.withLock { $0 } == ["model-b"])
    }
}
