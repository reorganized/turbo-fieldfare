import Foundation

enum AppModelLocation {
    /// The last model the user chose, honored before every default. One key,
    /// one line each way; the per-model settings file carries everything else.
    static var lastChosenPath: String? {
        get { UserDefaults.standard.string(forKey: "lastModelDirectory") }
        set { UserDefaults.standard.set(newValue, forKey: "lastModelDirectory") }
    }

    /// The launch model: the user's last choice if it still exists, else the
    /// pinned default.
    static func launchURL(fileExists: (String) -> Bool = FileManager.default.fileExists(atPath:)) -> URL {
        if let path = lastChosenPath, fileExists(path) {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return defaultURL()
    }

    static func defaultURL() -> URL {
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        return resolve(
            explicitURL: nil,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:))
    }

    static func resolve(explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        fileExists: (String) -> Bool) -> URL {
        if let explicitURL {
            return absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
        }
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return root.appendingPathComponent("scratch/gemma4.gturbo", isDirectory: true)
                .standardizedFileURL
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return root.appendingPathComponent("scratch/gemma4.gturbo", isDirectory: true)
                .standardizedFileURL
        }
        return applicationSupportURL
            .appendingPathComponent("TurboFieldfare", isDirectory: true)
            .appendingPathComponent("gemma4.gturbo", isDirectory: true)
            .standardizedFileURL
    }

    private static func absoluteURL(_ url: URL, relativeTo base: URL) -> URL {
        if url.path.hasPrefix("/") {
            return url.standardizedFileURL
        }
        return base.appendingPathComponent(url.path, isDirectory: true).standardizedFileURL
    }

    private static func packageRoot(startingAt start: URL,
                                    fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            let package = candidate.appendingPathComponent("Package.swift").path
            let appSources = candidate.appendingPathComponent(
                "Sources/TurboFieldfareApp/Mac", isDirectory: true).path
            if fileExists(package), fileExists(appSources) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
