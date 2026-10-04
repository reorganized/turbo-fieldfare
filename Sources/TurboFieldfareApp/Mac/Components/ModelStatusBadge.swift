import TurboFieldfareAppCore
import SwiftUI
import TurboFieldfareMacPresentation

struct ModelStatusBadge: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            statusDot
            Text(label)
                .font(.callout.weight(.semibold))
                .lineLimit(1)
                .help(help)
                .accessibilityLabel("Model")
                .accessibilityValue(model.installDescriptor.repoID)
                .accessibilityIdentifier(.hudStatus)
        }
    }

    /// The directory name, so any pack is labeled by what it actually is.
    private var label: String {
        model.modelPathText.split(separator: "/").last
            .map(String.init) ?? model.modelPathText
    }

    private var help: String { model.installDescriptor.repoID }

    @ViewBuilder
    private var statusDot: some View {
        switch model.presentation.severity {
        case .neutral: dot(.gray)
        case .active, .warning: dot(.orange)
        case .success: dot(.green)
        case .error: dot(.red)
        }
    }

    private func dot(_ color: Color) -> some View {
        Circle().fill(color).frame(width: 8, height: 8).accessibilityHidden(true)
    }
}
