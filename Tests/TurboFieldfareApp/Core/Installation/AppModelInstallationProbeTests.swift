import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelInstallationProbeTests {
    @Test func storageMetricsUseMetadataAndExcludeSymlinks() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("storage-metrics-\(UUID())", isDirectory: true)
        let nested = root.appendingPathComponent("nested", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("abc".utf8).write(to: root.appendingPathComponent("a"))
        try Data("12345".utf8).write(to: nested.appendingPathComponent("b"))
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("alias"),
            withDestinationURL: nested.appendingPathComponent("b"))

        let metrics = try AppModelStorageMetrics.measure(at: root)
        #expect(metrics.logicalBytes == 8)
        #expect(metrics.allocatedBytes >= metrics.logicalBytes)
        #expect(metrics.availableBytes != nil)
    }

    @Test func missingDirectoryIsMissing() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-missing-\(UUID().uuidString).gturbo")
        #expect(AppModelInstallationProbe.status(at: url) == .missing)
    }

    @Test func manifestWithoutFinalMetadataIsPartial() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbofieldfare-partial-\(UUID().uuidString).gturbo")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("{}".utf8).write(to: url.appendingPathComponent("manifest.json"))
        guard case .partial = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
    }

    @Test func validBoundedMetadataIsComplete() throws {
        let url = try makeCompleteModelInstall("probe")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(AppModelInstallationProbe.status(at: url) == .complete)
    }

    /// A pack the app never shipped is a complete install if its own
    /// manifest, layout, and receipt bind: the probe judges the directory,
    /// not the catalog.
    @Test func aQwenPackIsComplete() throws {
        let url = try makeCompleteModelInstall("qwen-probe",
                                               arch: .qwen3_6_35B_A3B,
                                               modelID: "test/qwen3.6-35b-a3b")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(AppModelInstallationProbe.status(at: url) == .complete)
    }

    /// The source snapshot hash is what ties the installed bytes to a known
    /// upstream state; a manifest without one is an install that cannot be
    /// vouched for, whatever else binds.
    @Test func manifestWithoutSnapshotHashIsPartial() throws {
        let url = try makeCompleteModelInstall("no-hash")
        defer { try? FileManager.default.removeItem(at: url) }
        let manifestURL = url.appendingPathComponent("manifest.json")
        var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
        manifest.removeValue(forKey: "sourceSnapshotHash")
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: manifestURL)
        try? FileManager.default.removeItem(at: url.appendingPathComponent("verified-install.json"))
        guard case .partial(let reason) = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
        #expect(reason.contains("source snapshot"))
    }

    @Test func loadFailureFixturePassesProbeButOmitsResidentWeights() throws {
        let output = try makeCompleteModelInstall("load-failure")
        defer { try? FileManager.default.removeItem(at: output) }
        #expect(AppModelInstallationProbe.status(at: output) == .complete)
        #expect(!FileManager.default.fileExists(
            atPath: output.appendingPathComponent("model_weights.bin").path))
    }

    @Test func receiptBoundToDifferentPathIsPartial() throws {
        let url = try makeCompleteModelInstall("wrong-path")
        defer { try? FileManager.default.removeItem(at: url) }
        let receiptURL = url.appendingPathComponent("verified-install.json")
        var receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as! [String: Any]
        receipt["modelDirectoryPath"] = "/different/model.gturbo"
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]).write(to: receiptURL)
        guard case .partial = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
    }

    /// Provenance is the verified receipt plus manifest hash binding, not a
    /// comparison against one descriptor: an install is judged by what binds
    /// in its own directory.
    @Test func differentCheckpointIsCompleteWhenReceiptBinds() throws {
        let url = try makeCompleteModelInstall("wrong-checkpoint")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(AppModelInstallationProbe.status(at: url) == .complete)
    }
}
