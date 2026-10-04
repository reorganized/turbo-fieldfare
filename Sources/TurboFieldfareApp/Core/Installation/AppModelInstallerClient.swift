import Foundation

public protocol AppModelInstallerClient: Sendable {
    /// The pack offered by default when no specific descriptor is given.
    var descriptor: AppModelInstallDescriptor { get }
    func checkInstallRequirement(_ descriptor: AppModelInstallDescriptor,
                                 outputDirectory: URL) throws -> AppModelInstallRequirement
    func install(_ descriptor: AppModelInstallDescriptor,
                 outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error>
    func discardPartialInstall(outputDirectory: URL) async throws
    func cancel()
}

extension AppModelInstallerClient {
    /// The single-descriptor shape the app had before the catalog: callers
    /// that install the pinned default keep reading naturally.
    func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement {
        try checkInstallRequirement(descriptor, outputDirectory: outputDirectory)
    }

    func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        install(.default, outputDirectory: outputDirectory)
    }
}
