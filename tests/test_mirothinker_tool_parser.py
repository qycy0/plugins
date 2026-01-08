import json
import sys
import types
import unittest
from dataclasses import dataclass

import pathlib

# Allow running tests from any working directory by ensuring repo root is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _install_stubs():
    """Install minimal stub modules so MirothinkerToolParser can be imported without vLLM."""

    # ---- vllm.logger ----
    vllm_logger = types.ModuleType("vllm.logger")

    class _Logger:
        def debug(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def exception(self, *args, **kwargs):
            return None

        def isEnabledFor(self, level: int) -> bool:
            return False

    def init_logger(name: str):
        return _Logger()

    vllm_logger.init_logger = init_logger

    # ---- vllm.entrypoints.chat_utils ----
    vllm_chat_utils = types.ModuleType("vllm.entrypoints.chat_utils")
    _id_counter = {"n": 0}

    def make_tool_call_id():
        _id_counter["n"] += 1
        return f"call_{_id_counter['n']}"

    vllm_chat_utils.make_tool_call_id = make_tool_call_id

    # ---- vllm.entrypoints.openai.protocol ----
    vllm_protocol = types.ModuleType("vllm.entrypoints.openai.protocol")

    @dataclass
    class FunctionCall:
        name: str
        arguments: str

    @dataclass
    class ToolCall:
        type: str
        function: FunctionCall

    @dataclass
    class ExtractedToolCallInformation:
        tools_called: bool
        tool_calls: list
        content: str | None

    @dataclass
    class DeltaFunctionCall:
        name: str | None = None
        arguments: str | None = None

        def model_dump(self, exclude_none: bool = False):
            d = {"name": self.name, "arguments": self.arguments}
            if exclude_none:
                return {k: v for k, v in d.items() if v is not None}
            return d

    @dataclass
    class DeltaToolCall:
        index: int
        type: str | None = None
        id: str | None = None
        function: dict | None = None

    @dataclass
    class DeltaMessage:
        content: str | None = None
        tool_calls: list | None = None

    @dataclass
    class _ToolFunction:
        name: str

    @dataclass
    class _Tool:
        function: _ToolFunction

    @dataclass
    class ChatCompletionRequest:
        tools: list | None = None
        tool_choice: str | None = None
        skip_special_tokens: bool | None = None

    vllm_protocol.ChatCompletionRequest = ChatCompletionRequest
    vllm_protocol.DeltaFunctionCall = DeltaFunctionCall
    vllm_protocol.DeltaMessage = DeltaMessage
    vllm_protocol.DeltaToolCall = DeltaToolCall
    vllm_protocol.ExtractedToolCallInformation = ExtractedToolCallInformation
    vllm_protocol.FunctionCall = FunctionCall
    vllm_protocol.ToolCall = ToolCall

    # ---- vllm.entrypoints.openai.tool_parsers.abstract_tool_parser ----
    vllm_tool_parser = types.ModuleType(
        "vllm.entrypoints.openai.tool_parsers.abstract_tool_parser"
    )

    class ToolParser:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def adjust_request(self, request):
            return request

    class ToolParserManager:
        @staticmethod
        def register_module(name: str, enabled: bool, cls):
            return None

    vllm_tool_parser.ToolParser = ToolParser
    vllm_tool_parser.ToolParserManager = ToolParserManager

    # ---- json_repair ----
    json_repair_mod = types.ModuleType("json_repair")

    def repair_json(s: str):
        # Lightweight repair for tests (simulate common json_repair fixes):
        # - remove trailing commas before } or ]
        # - allow single quotes for strings/keys (very approximate)
        # - convert Python literals True/False/None to JSON true/false/null (very approximate)
        try:
            json.loads(s)
            return s
        except Exception:
            fixed = s
            fixed = fixed.replace(",}", "}").replace(",]", "]")
            fixed = fixed.replace("True", "true").replace("False", "false").replace("None", "null")
            # naive single-quote to double-quote conversion (good enough for unit tests)
            fixed = fixed.replace("'", '"')
            try:
                json.loads(fixed)
                return fixed
            except Exception:
                return ""

    json_repair_mod.repair_json = repair_json

    # ---- partial_json_parser ---- (not used by correctness-first streaming, but imported)
    partial_json_parser_mod = types.ModuleType("partial_json_parser")

    def _pj_loads(s: str, flags=None):
        return json.loads(s)

    partial_json_parser_mod.loads = _pj_loads

    # ---- partial_json_parser.core.options.Allow ----
    partial_options_mod = types.ModuleType("partial_json_parser.core.options")

    class Allow:
        ALL = 0xFFFF
        STR = 0x0001

    partial_options_mod.Allow = Allow

    # ---- regex ---- (use stdlib re)
    import re as _re

    regex_mod = types.ModuleType("regex")
    regex_mod.compile = _re.compile
    regex_mod.DOTALL = _re.DOTALL

    sys.modules["vllm.logger"] = vllm_logger
    sys.modules["vllm.entrypoints.chat_utils"] = vllm_chat_utils
    sys.modules["vllm.entrypoints.openai.protocol"] = vllm_protocol
    sys.modules["vllm.entrypoints.openai.tool_parsers.abstract_tool_parser"] = vllm_tool_parser
    sys.modules["json_repair"] = json_repair_mod
    sys.modules["partial_json_parser"] = partial_json_parser_mod
    sys.modules["partial_json_parser.core.options"] = partial_options_mod
    sys.modules["regex"] = regex_mod

    return vllm_protocol, _Tool, _ToolFunction


class TestMirothinkerToolParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vllm_protocol, cls._Tool, cls._ToolFunction = _install_stubs()
        # Import after stubs are installed
        from tool_calling_plugin.MirothinkerToolParser import MirothinkerToolParser

        cls.Parser = MirothinkerToolParser

    def _make_request(self, tool_choice="auto"):
        tools = [
            self._Tool(self._ToolFunction("default.search")),
            self._Tool(self._ToolFunction("default.calc")),
        ]
        return self.vllm_protocol.ChatCompletionRequest(tools=tools, tool_choice=tool_choice)

    def test_non_streaming_basic(self):
        parser = self.Parser(tokenizer=None)
        req = self._make_request()
        model_output = (
            "hello\n"
            "<use_mcp_tool>\n"
            "<server_name>default</server_name>\n"
            "<tool_name>default.search</tool_name>\n"
            "<arguments>{\"q\": \"hi\"}</arguments>\n"
            "</use_mcp_tool>\n"
        )
        info = parser.extract_tool_calls(model_output, req)
        self.assertTrue(info.tools_called)
        self.assertEqual(len(info.tool_calls), 1)
        self.assertEqual(info.content, "hello\n")
        self.assertEqual(info.tool_calls[0].function.name, "default.search")
        self.assertEqual(json.loads(info.tool_calls[0].function.arguments), {"q": "hi"})

    def test_non_streaming_json_repair(self):
        parser = self.Parser(tokenizer=None)
        req = self._make_request()
        # trailing comma is invalid JSON but repairable
        model_output = (
            "<use_mcp_tool>"
            "<server_name>default</server_name>"
            "<tool_name>default.calc</tool_name>"
            "<arguments>{\"a\": 1,}</arguments>"
            "</use_mcp_tool>"
        )
        info = parser.extract_tool_calls(model_output, req)
        self.assertTrue(info.tools_called)
        self.assertEqual(len(info.tool_calls), 1)
        self.assertEqual(info.tool_calls[0].function.name, "default.calc")
        self.assertEqual(json.loads(info.tool_calls[0].function.arguments), {"a": 1})

    def test_non_streaming_invalid_json_unrepairable_skips_tool(self):
        parser = self.Parser(tokenizer=None)
        req = self._make_request()
        model_output = (
            "prefix"
            "<use_mcp_tool>"
            "<server_name>default</server_name>"
            "<tool_name>default.calc</tool_name>"
            "<arguments>{bad]</arguments>"
            "</use_mcp_tool>"
            "suffix"
        )
        info = parser.extract_tool_calls(model_output, req)
        # If unparseable, we should return the full content (do not truncate)
        self.assertFalse(info.tools_called)
        self.assertEqual(info.tool_calls, [])
        self.assertEqual(info.content, model_output)

    def test_non_streaming_missing_arguments_tag_no_match(self):
        parser = self.Parser(tokenizer=None)
        req = self._make_request()
        model_output = (
            "hello "
            "<use_mcp_tool>"
            "<server_name>default</server_name>"
            "<tool_name>default.search</tool_name>"
            # missing <arguments>...</arguments>
            "</use_mcp_tool>"
        )
        info = parser.extract_tool_calls(model_output, req)
        self.assertFalse(info.tools_called)
        self.assertEqual(info.tool_calls, [])
        self.assertEqual(info.content, model_output)

    def test_streaming_split_start_and_end_tags_correctness_first(self):
        parser = self.Parser(tokenizer=None)
        req = self._make_request()

        chunks = [
            "Hi ",
            "<use_m",
            "cp_tool><server_name>default</server_name>",
            "<tool_name>default.search</tool_name><arguments>{\"q\": \"he",
            "llo\"}</arguments></use_m",
            "cp_tool> Bye",
        ]

        previous = ""
        out_text = ""
        out_tool_calls = []
        for ch in chunks:
            current = previous + ch
            delta = parser.extract_tool_calls_streaming(
                previous, current, ch, [], [], [], req
            )
            if delta and delta.content:
                out_text += delta.content
            if delta and delta.tool_calls:
                out_tool_calls.extend(delta.tool_calls)
            previous = current

        self.assertEqual(out_text, "Hi  Bye")
        self.assertEqual(len(out_tool_calls), 1)
        tc = out_tool_calls[0]
        self.assertEqual(tc.index, 0)
        self.assertEqual(tc.type, "function")
        self.assertEqual(tc.function["name"], "default.search")
        self.assertEqual(json.loads(tc.function["arguments"]), {"q": "hello"})

    def test_streaming_repairable_json_single_quotes(self):
        parser = self.Parser(tokenizer=None)
        req = self._make_request()
        chunks = [
            "<use_mcp_tool><server_name>default</server_name>",
            "<tool_name>default.calc</tool_name><arguments>{'a': 1}</arguments>",
            "</use_mcp_tool>",
        ]
        previous = ""
        tool_calls = []
        for ch in chunks:
            current = previous + ch
            delta = parser.extract_tool_calls_streaming(previous, current, ch, [], [], [], req)
            if delta and delta.tool_calls:
                tool_calls.extend(delta.tool_calls)
            previous = current

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function["name"], "default.calc")
        self.assertEqual(json.loads(tool_calls[0].function["arguments"]), {"a": 1})

    def test_streaming_unrepairable_json_falls_back_to_text(self):
        parser = self.Parser(tokenizer=None)
        req = self._make_request()
        chunks = [
            "X",
            "<use_mcp_tool><server_name>default</server_name>",
            "<tool_name>default.calc</tool_name><arguments>{bad]</arguments></use_mcp_tool>",
            "Y",
        ]
        previous = ""
        out_text = ""
        out_tool_calls = []
        for ch in chunks:
            current = previous + ch
            delta = parser.extract_tool_calls_streaming(previous, current, ch, [], [], [], req)
            if delta and delta.content:
                out_text += delta.content
            if delta and delta.tool_calls:
                out_tool_calls.extend(delta.tool_calls)
            previous = current

        # Should not emit tool_calls; tool block should be treated as plain text
        self.assertEqual(out_tool_calls, [])
        self.assertIn("<use_mcp_tool>", out_text)
        self.assertTrue(out_text.startswith("X"))
        self.assertTrue(out_text.endswith("Y"))

    def test_streaming_two_tools(self):
        parser = self.Parser(tokenizer=None)
        req = self._make_request()

        ch = (
            "A"
            "<use_mcp_tool><server_name>default</server_name><tool_name>default.search</tool_name>"
            "<arguments>{\"q\":\"x\"}</arguments></use_mcp_tool>"
            "B"
            "<use_mcp_tool><server_name>default</server_name><tool_name>default.calc</tool_name>"
            "<arguments>{\"a\":2}</arguments></use_mcp_tool>"
            "C"
        )
        previous = ""
        out_text = ""
        out_tool_calls = []
        # send in two deltas to cover "multiple blocks in one chunk" too
        for part in [ch[: len(ch) // 2], ch[len(ch) // 2 :]]:
            current = previous + part
            delta = parser.extract_tool_calls_streaming(
                previous, current, part, [], [], [], req
            )
            if delta and delta.content:
                out_text += delta.content
            if delta and delta.tool_calls:
                out_tool_calls.extend(delta.tool_calls)
            previous = current

        self.assertEqual(out_text, "ABC")
        self.assertEqual(len(out_tool_calls), 2)
        self.assertEqual(out_tool_calls[0].index, 0)
        self.assertEqual(out_tool_calls[1].index, 1)
        self.assertEqual(out_tool_calls[0].function["name"], "default.search")
        self.assertEqual(out_tool_calls[1].function["name"], "default.calc")

    def test_streaming_tool_choice_none_passthrough(self):
        parser = self.Parser(tokenizer=None)
        req = self._make_request(tool_choice="none")
        previous = ""
        ch = "<use_mcp_tool><server_name>default</server_name></use_mcp_tool>"
        current = ch
        delta = parser.extract_tool_calls_streaming(
            previous, current, ch, [], [], [], req
        )
        # correctness-first: when tool_choice is none, it should not suppress tags
        self.assertIsNotNone(delta)
        self.assertEqual(delta.content, ch)
        self.assertTrue(delta.tool_calls is None or delta.tool_calls == [])


if __name__ == "__main__":
    unittest.main()


