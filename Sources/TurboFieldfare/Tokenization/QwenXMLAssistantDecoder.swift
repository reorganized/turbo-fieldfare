import Foundation

/// Streaming decoder for the Qwen3.6 tool-call markup that the model's own
/// chat template asks for when tools are present
/// (`scratch/qwen36.gturbo/tokenizer/chat_template.jinja`):
///
///     <tool_call>
///     <function=NAME>
///     <parameter=KEY>
///     VALUE
///     </parameter>
///     </function>
///     </tool_call>
///
/// The grammar follows the vLLM / mlx-lm `qwen3_coder` parsers. `<tool_call>`
/// and `</tool_call>` are ordinary (non-special) tokens in this vocabulary and
/// the tokenizer aliases every Gemma tool/channel marker to `<eos>` for
/// `chatStyle == .qwenImStart`, so the token-ID machinery in
/// `StructuredAssistantDecoder` never fires. This decoder scans the content
/// deltas instead: text outside a call flows through (holding back a partial
/// opener and trailing whitespace), an opener switches to buffering, and the
/// block is parsed once its closer arrives.
///
/// Unlike the Gemma path, failures degrade to plain content (finish reason
/// "stop") instead of failing the request: the markup is only advice the
/// template gives the model, and a malformed block was already plain text
/// before this decoder existed.
public final class QwenXMLAssistantDecoder: @unchecked Sendable, AssistantStreamDecoding {
    static let toolCallTag = "<tool_call>"
    static let functionTag = "<function="
    static let functionEndTag = "</function>"
    static let toolCallEndTag = "</tool_call>"
    static let maximumBytes = GemmaToolCallParser.maximumBytes

    /// Allowed tool name → parameters whose schema declares only non-string
    /// types. Those values are JSON-decoded; every other value stays the
    /// literal text the model wrote, so a string such as `"42"` or file
    /// content is never retyped.
    private let typedParameters: [String: Set<String>]
    private let idGenerator: @Sendable () -> String
    private var hold = ""
    private var inCall = false
    private var callBuffer = ""
    private var closer = ""
    /// Whitespace withheld before the open call; restored if it degrades.
    private var separator = ""
    private var emittedCalls = 0

    public init(tools: [GFTokenizer.FunctionDefinition],
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.typedParameters = Dictionary(
            tools.map { ($0.name, Self.typedParameterNames($0.parameters)) },
            uniquingKeysWith: { first, _ in first })
        self.idGenerator = idGenerator
    }

    public func consume(tokenID: Int32, delta: String) -> [StructuredAssistantEvent] {
        guard !delta.isEmpty else { return [] }
        guard inCall else { return stream(delta) }
        callBuffer += delta
        return scanCall(newCharacters: delta.count)
    }

    /// Flush at the stop boundary. Trailing whitespace is dropped; a partial
    /// opener is prose. A call still open is parsed as-is, so a block missing
    /// only its final `</tool_call>` still counts; otherwise it degrades.
    public func consumeTail(_ text: String) -> [StructuredAssistantEvent] {
        var events = consume(tokenID: -1, delta: text)
        if inCall {
            let block = callBuffer
            callBuffer = ""
            inCall = false
            events += emit(block: block)
        } else {
            let rest = Self.trimmingTrailingWhitespace(hold)
            hold = ""
            if !rest.isEmpty { events.append(.content(rest)) }
        }
        return events
    }

    public func finish() throws {
        // `consumeTail` flushes everything at the stop boundary.
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }

    // MARK: - Streaming state machine

    private func stream(_ delta: String) -> [StructuredAssistantEvent] {
        hold += delta
        let openers = [hold.range(of: Self.toolCallTag), hold.range(of: Self.functionTag)]
        if let opener = openers.compactMap({ $0 }).min(by: { $0.lowerBound < $1.lowerBound }) {
            // Whitespace between prose and a call (and between two calls) is
            // template separator, not content.
            let prefix = hold[..<opener.lowerBound]
            let before = Self.trimmingTrailingWhitespace(prefix)
            separator = String(prefix.dropFirst(before.count))
            callBuffer = String(hold[opener.lowerBound...])
            hold = ""
            inCall = true
            closer = callBuffer.hasPrefix(Self.toolCallTag) ? Self.toolCallEndTag : Self.functionEndTag
            return (before.isEmpty ? [] : [.content(before)])
                + scanCall(newCharacters: callBuffer.count)
        }
        let keep = Self.holdbackLength(hold)
        guard keep < hold.count else { return [] }
        let emit = String(hold.dropLast(keep))
        hold = String(hold.suffix(keep))
        return [.content(emit)]
    }

    /// Look for the closer only where the newest characters could complete
    /// it, so a long argument payload is not rescanned on every token.
    private func scanCall(newCharacters: Int) -> [StructuredAssistantEvent] {
        guard callBuffer.utf8.count <= Self.maximumBytes else {
            // Too large to be a call: hand back what the model wrote.
            let text = separator + callBuffer
            callBuffer = ""
            separator = ""
            inCall = false
            return [.content(text)]
        }
        var searchStart = callBuffer.index(
            callBuffer.endIndex,
            offsetBy: -(newCharacters + closer.count),
            limitedBy: callBuffer.startIndex) ?? callBuffer.startIndex
        while let end = callBuffer.range(of: closer, range: searchStart..<callBuffer.endIndex) {
            // `</tool_call>` inside an argument (code that mentions the tag)
            // is not the closer; the real one follows `</function>`.
            guard closer == Self.functionEndTag
                    || Self.endsWithFunctionClose(callBuffer[..<end.lowerBound]) else {
                searchStart = end.upperBound
                continue
            }
            let block = String(callBuffer[..<end.upperBound])
            let trailing = String(callBuffer[end.upperBound...])
            callBuffer = ""
            inCall = false
            return emit(block: block) + (trailing.isEmpty ? [] : stream(trailing))
        }
        return []
    }

    /// A parsed block becomes a tool call; anything else is returned verbatim,
    /// with its withheld leading whitespace, as content rather than dropped.
    private func emit(block: String) -> [StructuredAssistantEvent] {
        defer { separator = "" }
        guard let call = parse(block, id: idGenerator()) else {
            let text = separator + block
            return text.isEmpty ? [] : [.content(text)]
        }
        emittedCalls += 1
        return [.toolCall(call)]
    }

    private static func endsWithFunctionClose(_ text: Substring) -> Bool {
        guard let last = text.lastIndex(where: { !$0.isWhitespace }) else { return false }
        return text[...last].hasSuffix(functionEndTag)
    }

    /// Characters at the end of `text` that must wait for the next delta: a
    /// strict prefix of an opener tag, plus any whitespace right before it.
    private static func holdbackLength(_ text: String) -> Int {
        let characters = Array(text)
        var opener = 0
        for tag in [Array(toolCallTag), Array(functionTag)] {
            for length in stride(from: min(characters.count, tag.count - 1), through: 1, by: -1)
            where characters.suffix(length).elementsEqual(tag.prefix(length)) {
                opener = max(opener, length)
                break
            }
        }
        let whitespace = characters.dropLast(opener).reversed().prefix(while: \.isWhitespace).count
        return opener + whitespace
    }

    private static func trimmingTrailingWhitespace(_ text: some StringProtocol) -> String {
        guard let last = text.lastIndex(where: { !$0.isWhitespace }) else { return "" }
        return String(text[...last])
    }

    // MARK: - Block parsing

    /// Parse one block (opener through closer). Returns nil on malformed
    /// grammar, an unknown tool, or a duplicate parameter.
    func parse(_ block: String, id: String) -> ParsedToolCall? {
        var scanner = BlockScanner(block)
        scanner.skipWhitespace()
        _ = scanner.consume(Self.toolCallTag)
        scanner.skipWhitespace()
        guard scanner.consume(Self.functionTag),
              let name = scanner.take(through: ">"),
              let typed = typedParameters[name] else { return nil }
        var arguments: [String: JSONValue] = [:]
        while true {
            scanner.skipWhitespace()
            if scanner.consume(Self.functionEndTag) { break }
            guard scanner.consume("<parameter="),
                  let key = scanner.take(through: ">"),
                  GemmaToolCallParser.isRepresentableObjectKey(key),
                  arguments[key] == nil,
                  let raw = scanner.takeParameterValue() else { return nil }
            arguments[key] = Self.value(raw, typed: typed.contains(key))
        }
        scanner.skipWhitespace()
        _ = scanner.consume(Self.toolCallEndTag)
        scanner.skipWhitespace()
        guard scanner.isAtEnd else { return nil }
        let object = JSONValue.object(arguments)
        guard let json = try? object.encoded() else { return nil }
        return ParsedToolCall(id: id, name: name, arguments: object, argumentsJSON: json)
    }

    /// The template writes `<parameter=K>\n` + value + `\n</parameter>`, so
    /// exactly one newline is stripped from each side; inner whitespace
    /// (indentation, blank lines) is the value. Non-string parameters are
    /// JSON-decoded, falling back to the text.
    static func value(_ raw: String, typed: Bool) -> JSONValue {
        var text = Substring(raw)
        if text.first == "\n" { text = text.dropFirst() }
        if text.last == "\n" { text = text.dropLast() }
        if typed, let decoded = try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)) {
            return decoded
        }
        return .string(String(text))
    }

    /// Top-level parameter names whose schema `type` (or `anyOf`/`oneOf`
    /// branch types) is declared and excludes `string`.
    static func typedParameterNames(_ parameters: JSONValue) -> Set<String> {
        guard case .object(let schema) = parameters,
              case .object(let properties) = schema["properties"] else { return [] }
        return Set(properties.compactMap { name, property in
            let types = declaredTypes(property)
            return !types.isEmpty && !types.contains("string") ? name : nil
        })
    }

    private static func declaredTypes(_ schema: JSONValue) -> Set<String> {
        guard case .object(let object) = schema else { return [] }
        var types: Set<String> = []
        switch object["type"] {
        case .string(let type): types.insert(type)
        case .array(let list): for case .string(let type) in list { types.insert(type) }
        default: break
        }
        for key in ["anyOf", "oneOf"] {
            if case .array(let branches) = object[key] {
                for branch in branches { types.formUnion(declaredTypes(branch)) }
            }
        }
        return types
    }
}

/// Minimal forward scanner over one call block.
private struct BlockScanner {
    let text: String
    private var index: String.Index

    init(_ text: String) {
        self.text = text
        self.index = text.startIndex
    }

    var isAtEnd: Bool { index == text.endIndex }

    mutating func skipWhitespace() {
        while index < text.endIndex, text[index].isWhitespace {
            index = text.index(after: index)
        }
    }

    mutating func consume(_ literal: String) -> Bool {
        guard text[index...].hasPrefix(literal) else { return false }
        index = text.index(index, offsetBy: literal.count)
        return true
    }

    /// The text up to `terminator` (consumed, not returned); nil if absent.
    mutating func take(through terminator: String) -> String? {
        guard let end = text.range(of: terminator, range: index..<text.endIndex) else { return nil }
        let taken = String(text[index..<end.lowerBound])
        index = end.upperBound
        return taken
    }

    /// A parameter value, ending at the first `</parameter>` that is followed
    /// by the next `<parameter=` or by `</function>`, so a value may itself
    /// contain `</parameter>`.
    mutating func takeParameterValue() -> String? {
        var from = index
        while let end = text.range(of: "</parameter>", range: from..<text.endIndex) {
            let rest = text[end.upperBound...].drop(while: \.isWhitespace)
            if rest.hasPrefix("<parameter=") || rest.hasPrefix(QwenXMLAssistantDecoder.functionEndTag) {
                let taken = String(text[index..<end.lowerBound])
                index = end.upperBound
                return taken
            }
            from = end.upperBound
        }
        return nil
    }
}
