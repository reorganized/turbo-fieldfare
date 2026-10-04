import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareDecodeService

/// The service is the process the system kills. A 262,144-token KV is charged
/// in full the moment a command buffer binds it — one full-attention layer over
/// sixteen rows already costs 1,085 MiB — so a load this host cannot back does
/// not fail slowly and report itself: it disappears mid-load, and the app can
/// only show a lost connection. Every case here is a load that must be answered
/// before the runner is built.
@Suite struct DecodeLoadAdmissionTests {

    @Test(arguments: [24, 32])
    func cacheGrowthRejects128KOnEightGB(slots: Int) throws {
        let refusal = try #require(DecodeLoadAdmission.refusal(maxContextTokens: 131_072,
            hostMemoryBytes: 8 << 30, environment: [:], expertCacheSlots: slots))
        #expect(refusal.contains("16 GB"))
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: 131_072,
            hostMemoryBytes: 16 << 30, environment: [:], expertCacheSlots: slots) == nil)
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: 131_072,
            hostMemoryBytes: 8 << 30,
            environment: [DecodeLoadAdmission.overrideEnvironmentKey: "1"],
            expertCacheSlots: slots) == nil)
    }

    private let eightGigabyteHost: UInt64 = 8_589_934_592
    private let twentyFourGigabyteHost: UInt64 = 25_769_803_776

    @Test func acontextThisHostCanBackIsAdmitted() {
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: 8_192,
                                            hostMemoryBytes: eightGigabyteHost,
                                            environment: [:]) == nil)
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: 262_144,
                                            hostMemoryBytes: twentyFourGigabyteHost,
                                            environment: [:]) == nil)
    }

    @Test func acontextThisHostCannotBackIsRefusedWithTheNeed() throws {
        let refusal = try #require(
            DecodeLoadAdmission.refusal(maxContextTokens: 262_144,
                                        hostMemoryBytes: eightGigabyteHost,
                                        environment: [:]))
        #expect(refusal.contains("262,144"))
        #expect(refusal.contains("16 GB"))
        #expect(refusal.contains("8 GB"))
    }

    /// The measurement the admission rule came from is only reachable by
    /// running a context the rule refuses, so the override exists — explicitly,
    /// and for nothing else.
    @Test func theoverrideAdmitsAnUnbackedContext() {
        #expect(DecodeLoadAdmission.refusal(
            maxContextTokens: 262_144,
            hostMemoryBytes: eightGigabyteHost,
            environment: [DecodeLoadAdmission.overrideEnvironmentKey: "1"]) == nil)
    }

    @Test(arguments: ["0", "", "true", "yes"])
    func theoverrideIsExactlyOneAndNothingElse(_ value: String) {
        #expect(DecodeLoadAdmission.refusal(
            maxContextTokens: 262_144,
            hostMemoryBytes: eightGigabyteHost,
            environment: [DecodeLoadAdmission.overrideEnvironmentKey: value]) != nil,
            "\(value) was treated as the override")
    }

    /// The model ceiling is not a memory question, so the diagnostic override
    /// must not reach it: a position beyond `max_position_embeddings` is
    /// invalid on a 512 GB machine too.
    @Test func themodelCeilingIsRefusedEvenWithTheOverrideSet() throws {
        let ceiling = ArchConfig.gemma4_26B_A4B.maxPositionEmbeddings
        let refusal = try #require(DecodeLoadAdmission.refusal(
            maxContextTokens: ceiling + 1,
            hostMemoryBytes: twentyFourGigabyteHost,
            environment: [DecodeLoadAdmission.overrideEnvironmentKey: "1"]))
        #expect(refusal.contains("262145"))
        #expect(refusal.contains("262144"))
        #expect(DecodeLoadAdmission.refusal(maxContextTokens: ceiling,
                                            hostMemoryBytes: twentyFourGigabyteHost,
                                            environment: [:]) == nil)
    }

    /// A different checkpoint means a different ceiling and a different KV
    /// geometry: a Qwen hybrid pack is refused at its own 262,144 ceiling, not
    /// at whatever memory the Gemma profile computed, and a Gemma-sized
    /// context stays valid for it.
    @Test func theqwenProfileIsJudgedByItsOwnCeilingAndMemory() throws {
        let qwen = ArchConfig.qwen3_6_35B_A3B
        #expect(qwen.maxPositionEmbeddings == 262_144)
        #expect(DecodeLoadAdmission.refusal(
            maxContextTokens: 262_144,
            hostMemoryBytes: twentyFourGigabyteHost,
            environment: [:], config: qwen) == nil)

        let refusal = try #require(DecodeLoadAdmission.refusal(
            maxContextTokens: qwen.maxPositionEmbeddings + 1,
            hostMemoryBytes: UInt64.max,
            environment: [:], config: qwen))
        #expect(refusal.contains("262145"))
        // The hybrid KV is cheaper than Gemma's: the same 128K context that an
        // 8 GB Mac refuses for Gemma can be backed for Qwen.
        #expect(DecodeLoadAdmission.refusal(
            maxContextTokens: 131_072,
            hostMemoryBytes: eightGigabyteHost,
            environment: [:], config: qwen) == nil
            || refusal.contains("262,145"))
    }
}
