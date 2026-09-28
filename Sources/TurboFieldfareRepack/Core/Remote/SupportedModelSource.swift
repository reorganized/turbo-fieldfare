import Foundation

/// Pinned, installer-validated source snapshots. Adding an entry means the
/// importer has been validated against that upload; `SourceFingerprint`
/// mirrors the index SHA-256 pins.
public enum SupportedModelSource {
    public struct Entry: Sendable, Equatable {
        public let id: String
        public let displayName: String
        public let repoID: String
        public let revision: String
        public let sourceIndexSHA256: String
        public let approximateDownloadBytes: UInt64
        public let installedBytes: UInt64
        public let reserveBytes: UInt64

        public func installOptions(outputDirectory: URL,
                                   overwrite: Bool,
                                   token: String?,
                                   resume: Bool = false)
            -> RemoteStreamingRepackOptions {
            RemoteStreamingRepackOptions(
                repoID: repoID,
                revision: revision,
                outputDir: outputDirectory.path,
                token: token,
                requireKnownSource: true,
                minFreeReserveBytes: reserveBytes,
                overwrite: overwrite,
                resume: resume)
        }
    }

    public static let gemma4 = Entry(
        id: "gemma4",
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: 1_073_741_824)

    public static let qwen36 = Entry(
        id: "qwen36",
        displayName: "Qwen3.6 35B-A3B 4-bit",
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256:
            "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        approximateDownloadBytes: 20_430_000_000,
        installedBytes: 19_600_000_000,
        reserveBytes: 2_147_483_648)

    public static let all: [Entry] = [gemma4, qwen36]
    public static let defaultEntry = gemma4

    public static func entry(id: String) -> Entry? {
        all.first { $0.id == id }
    }

    // Backward-compatible accessors for the default source. The vision-pack
    // installer is Gemma-only and keeps using these.
    public static let displayName = gemma4.displayName
    public static let repoID = gemma4.repoID
    public static let revision = gemma4.revision
    public static let sourceIndexSHA256 = gemma4.sourceIndexSHA256
    public static let approximateDownloadBytes = gemma4.approximateDownloadBytes
    public static let installedBytes = gemma4.installedBytes
    public static let reserveBytes = gemma4.reserveBytes

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        gemma4.installOptions(outputDirectory: outputDirectory,
                              overwrite: overwrite,
                              token: token,
                              resume: resume)
    }
}
