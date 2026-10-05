import Foundation
import Testing
@testable import TurboFieldfare

/// Qwen3.6 always renders with thinking enabled; `QwenThinkingFilter` keeps
/// the reasoning out of visible text. The template cases load the installed
/// tokenizer and skip themselves when `scratch/qwen36.gturbo` is missing.
@Suite("Qwen thinking")
struct QwenThinkingTests {
    private static let modelDirectory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("scratch/qwen36.gturbo")
    private static let installed = FileManager.default.fileExists(
        atPath: modelDirectory.appendingPathComponent("tokenizer/chat_template.jinja").path)

    private static func tokenizer() async throws -> GFTokenizer {
        try await GFTokenizer.load(forModelDirectory: modelDirectory)
    }

    private static let bash = GFTokenizer.FunctionDefinition(
        name: "bash",
        description: "Run a command.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object(["command": .object(["type": .string("string")])]),
        ]))

    @Test("Reasoning and the blank lines after it are hidden, split anywhere")
    func filterHidesReasoning() {
        var filter = QwenThinkingFilter(active: true)
        let pieces = ["Let me", " think.\n</th", "ink>", "\n\n", "Hi", " there"]
        #expect(pieces.map { filter.push($0) }.joined() == "Hi there")
        var sameDelta = QwenThinkingFilter(active: true)
        #expect(sameDelta.push("x</think>\n\nAnswer") == "Answer")
        var indented = QwenThinkingFilter(active: true)
        #expect(indented.push("x</think>\n\n    code()") == "    code()")
    }

    @Test("An inactive filter passes text through")
    func inactiveFilterPassesThrough() {
        var filter = QwenThinkingFilter(active: false)
        #expect(filter.push("\n</think> a") == "\n</think> a")
    }

    @Test("Prompts open a think block and activate the filter",
          .enabled(if: installed))
    func promptsOpenThinkBlock() async throws {
        let tok = try await Self.tokenizer()
        let user = [GFTokenizer.Message(role: .user, content: "hi")]
        for prompt in [try tok.encodeChat(messages: user),
                       try tok.encodeToolChat(messages: user, tools: [Self.bash])] {
            #expect(tok.decode(prompt, skipSpecialTokens: false)
                .hasSuffix("<|im_start|>assistant\n<think>\n"))
            var filter = QwenThinkingFilter(tokenizer: tok, promptIds: prompt)
            #expect(filter.push("plan</think>\n\nok") == "ok")
        }
    }

    @Test("A text continuation is the template's next turn and fits the envelope",
          .enabled(if: installed))
    func textContinuation() async throws {
        let tok = try await Self.tokenizer()
        let bridge = tok.encodeTextContinuation(userContent: "x")
        #expect(bridge.first == tok.endOfTurnID)
        #expect(tok.decode(bridge, skipSpecialTokens: false)
            == "<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n")
        #expect(bridge.count <= ConversationGenerationReserve.turnEnvelope)
    }

    @Test("The tool-result bridge is the tail of the full render",
          .enabled(if: installed))
    func toolResultBridge() async throws {
        let tok = try await Self.tokenizer()
        let user = GFTokenizer.Message(role: .user, content: "list files")
        let assistant = GFTokenizer.Message(
            role: .assistant, content: nil,
            toolCalls: [GFTokenizer.HistoricalToolCall(
                id: "call_1", name: "bash", arguments: .object(["command": .string("ls")]))])
        let result = GFTokenizer.Message(role: .tool, content: "a\nb", toolCallID: "call_1")
        let incoming = [user, assistant, result]
        let bridge = try tok.encodeToolResultContinuation(
            cachedMessages: [user], assistant: assistant,
            incomingMessages: incoming, tools: [Self.bash])
        let full = try tok.encodeToolChat(messages: incoming, tools: [Self.bash])
        #expect(Array(full.suffix(bridge.count)) == bridge)
        #expect(tok.decode(bridge, skipSpecialTokens: false) == "<|im_end|>\n<|im_start|>user\n"
            + "<tool_response>\na\nb\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")
    }
}
