import Foundation
import Testing
@testable import TurboFieldfare

/// Model-free coverage for `QwenXMLAssistantDecoder`, the text-level decoder
/// for the Qwen3.6 `<tool_call>` / `<function=...>` markup. Deltas are fed in
/// small chunks to exercise the streaming holdback; adjacent content events
/// are merged because chunking splits them.
@Suite("Qwen XML assistant decoder")
struct QwenXMLAssistantDecoderTests {
    private static let tools = [GFTokenizer.FunctionDefinition(
        name: "bash",
        description: "",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "command": .object(["type": .string("string")]),
                "timeout": .object(["type": .string("integer")]),
                "flags": .object(["type": .string("array")]),
                "limit": .object(["anyOf": .array([
                    .object(["type": .string("integer")]),
                    .object(["type": .string("null")]),
                ])]),
            ]),
        ]))]

    private static func decoder() -> QwenXMLAssistantDecoder {
        QwenXMLAssistantDecoder(tools: tools, idGenerator: { "call_test" })
    }

    private func run(_ text: String, width: Int = 3, tail: String = "") -> [StructuredAssistantEvent] {
        let decoder = Self.decoder()
        var events: [StructuredAssistantEvent] = []
        var index = text.startIndex
        while index < text.endIndex {
            let end = text.index(index, offsetBy: width, limitedBy: text.endIndex) ?? text.endIndex
            events += decoder.consume(tokenID: -1, delta: String(text[index..<end]))
            index = end
        }
        events += decoder.consumeTail(tail)
        return Self.merged(events)
    }

    private static func merged(_ events: [StructuredAssistantEvent]) -> [StructuredAssistantEvent] {
        var result: [StructuredAssistantEvent] = []
        for event in events {
            if case .content(let text) = event, case .content(let previous)? = result.last {
                result[result.count - 1] = .content(previous + text)
            } else {
                result.append(event)
            }
        }
        return result
    }

    private static func call(_ arguments: [String: JSONValue]) -> StructuredAssistantEvent {
        .toolCall(ParsedToolCall(
            id: "call_test",
            name: "bash",
            arguments: .object(arguments),
            argumentsJSON: try! JSONValue.object(arguments).encoded()))
    }

    /// The exact shape the chat template renders for a historical call.
    private static func block(_ parameters: [(String, String)], name: String = "bash") -> String {
        "<tool_call>\n<function=\(name)>\n"
            + parameters.map { "<parameter=\($0)>\n\($1)\n</parameter>\n" }.joined()
            + "</function>\n</tool_call>"
    }

    @Test("Call inside prose is promoted; separator whitespace is dropped")
    func callInsideProse() {
        let text = "Let me check.\n\n" + Self.block([("command", "ls -la")]) + "\ndone."
        #expect(run(text) == [
            .content("Let me check."),
            Self.call(["command": .string("ls -la")]),
            .content("\ndone."),
        ])
    }

    @Test("One-character deltas give the same result")
    func splitTags() {
        let text = "ok\n" + Self.block([("command", "echo hi")])
        #expect(run(text, width: 1) == [
            .content("ok"),
            Self.call(["command": .string("echo hi")]),
        ])
    }

    @Test("Schema types decide JSON decoding; string values stay text")
    func schemaTypedValues() {
        let text = Self.block([
            ("command", "42"),
            ("timeout", "300"),
            ("flags", #"["-a", "-l"]"#),
            ("limit", "7"),
        ])
        #expect(run(text) == [Self.call([
            "command": .string("42"),
            "timeout": .integer(300),
            "flags": .array([.string("-a"), .string("-l")]),
            "limit": .integer(7),
        ])])
    }

    @Test("String values keep indentation and inner blank lines")
    func whitespacePreserved() {
        let value = "    indented()\n\n  next\n"
        #expect(run(Self.block([("command", value)])) == [
            Self.call(["command": .string(value)]),
        ])
    }

    @Test("Unknown tool degrades to verbatim content")
    func unknownToolDegrades() {
        let block = Self.block([("path", "/x")], name: "edit")
        let decoder = Self.decoder()
        let events = Self.merged(decoder.consume(tokenID: -1, delta: block) + decoder.consumeTail(""))
        #expect(events == [.content(block)])
        #expect(!decoder.hasToolCalls)
    }

    @Test("Malformed block degrades once; text after it is not repeated")
    func malformedDegrades() {
        let block = "<tool_call>\n<function=bash\n<parameter=command>x</parameter>\n</function>\n</tool_call>"
        #expect(run(block + " after") == [.content(block + " after")])
    }

    @Test("Degraded text keeps the whitespace before the tag")
    func degradedKeepsSeparator() {
        #expect(run("Use <tool_call> tags.") == [.content("Use <tool_call> tags.")])
        let text = "A\n\n<tool_call>bogus</tool_call>\n\nB"
        #expect(run(text) == [.content(text)])
    }

    @Test("Closing tags inside a value do not end the call")
    func closersInsideValue() {
        let value = "echo '</tool_call>' '</parameter>'"
        #expect(run(Self.block([("command", value)]), width: 1) == [
            Self.call(["command": .string(value)]),
        ])
    }

    @Test("A block missing only its final </tool_call> still parses at stop")
    func missingCloserAtStop() {
        let text = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n"
        #expect(run(text) == [Self.call(["command": .string("ls")])])
    }

    @Test("Truncated call at stop degrades to content")
    func truncatedCallDegrades() {
        let text = "<tool_call>\n<function=bash>\n<parameter=command>\nls"
        #expect(run(text) == [.content(text)])
    }

    @Test("Partial opener at end of stream is content")
    func partialOpenerFlushed() {
        #expect(run("answer is 4 <tool_") == [.content("answer is 4 <tool_")])
    }

    @Test("Two calls separated by the template newline")
    func multipleCalls() {
        let text = Self.block([("command", "a")]) + "\n" + Self.block([("command", "b")])
        #expect(run(text) == [
            Self.call(["command": .string("a")]),
            Self.call(["command": .string("b")]),
        ])
    }

    @Test("A bare <function=...> block closes at </function>")
    func bareFunctionBlock() {
        let text = "<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>"
        #expect(run(text, tail: "\n") == [Self.call(["command": .string("pwd")])])
    }

    @Test("Oversized call buffer degrades instead of growing unbounded")
    func oversizedDegrades() {
        let payload = String(repeating: "x", count: QwenXMLAssistantDecoder.maximumBytes + 1)
        let events = run(Self.block([("command", payload)]), width: 4096)
        guard case .content(let emitted) = events.first else {
            Issue.record("expected degraded content")
            return
        }
        #expect(emitted.hasPrefix("<tool_call>"))
        #expect(events.count == 1)
    }
}
