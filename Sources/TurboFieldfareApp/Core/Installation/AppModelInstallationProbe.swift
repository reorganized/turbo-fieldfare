import Foundation
import TurboFieldfare

public enum AppModelInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
}

public enum AppModelInstallationProbe {
    public static func status(
        at directory: URL
    ) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }

        // The arch is whatever the manifest records; `expecting` only governs
        // legacy nil-field manifests, which stay Gemma-only by design. Install
        // provenance is the verified receipt plus the manifest hash binding,
        // not a comparison against one descriptor: a pack the descriptor never
        // knew about is still a complete install if its own receipt checks out.
        do {
            let manifest = try ManifestReader.load(directoryURL: directory, expecting: .gemma4_26B_A4B)
            guard let source = manifest.sourceSnapshotHash, !source.isEmpty else {
                return .partial("installed manifest carries no source snapshot hash")
            }
            let layout = directory.appendingPathComponent("packed_experts/layout.json")
            guard FileManager.default.fileExists(atPath: layout.path) else {
                return .partial("packed_experts/layout.json is missing")
            }
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }
}
