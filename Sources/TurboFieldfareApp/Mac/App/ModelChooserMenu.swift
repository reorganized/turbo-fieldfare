import TurboFieldfareAppCore
import SwiftUI

/// Chooses an installed `.gturbo` pack by directory.
///
/// `AppModel.reviewExistingModelLocation` owns every switch mechanic
/// (validate, persist, rebind history, re-probe); this only picks the
/// directory. It disables itself whenever that call would no-op — an offered
/// control that silently does nothing is worse than a missing one.
struct ModelChooserMenu: View {
    let model: AppModel
    @State private var isPresentingChooser = false

    var body: some View {
        Group {
            if isStripButton {
                Button {
                    isPresentingChooser = true
                } label: {
                    Image(systemName: "shippingbox")
                        .font(.system(size: 16, weight: .regular))
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(StripControlButtonStyle())
                .foregroundStyle(model.canChooseModel ? Color.primary : Color.secondary.opacity(0.5))
                .help("Choose Model…")
                .accessibilityLabel("Choose Model…")
            } else {
                Button("Choose Model…") {
                    isPresentingChooser = true
                }
            }
        }
        .disabled(!model.canChooseModel)
        .fileImporter(
            isPresented: $isPresentingChooser,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let url = urls.first else { return }
            Task { await model.reviewExistingModelLocation(url) }
        }
    }

    /// The status row renders it as a plain strip control; the Model menu
    /// renders it as a labeled item. One action behind both.
    init(stripButton: Bool = false, model: AppModel) {
        self.model = model
        self.isStripButton = stripButton
    }

    private let isStripButton: Bool
}

extension AppModel {
    public var canChooseModel: Bool { canChangeModelLocation }
}
