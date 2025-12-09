"""
Tool parser plugin for vLLM for MiroThinker MCP format to compatible with the tool calling interface of openai.
MCP format:
    <use_mcp_tool>
        <server_name>server name</server_name>
        <tool_name>tool name</tool_name>
        <arguments>
        {...}
        </arguments>
    </use_mcp_tool>
in chat template, we advise model to use `default` as server name. but if model can resolve the server name from tool name and use it in server name.
we also can resolve the tool name from tool name lists when using mcp server.
"""
import json
from collections.abc import Sequence

import partial_json_parser
import regex as re
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,ToolParserManager
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class MirothinkerToolParser(ToolParser):
    """
    Tool parser for MiroThinker MCP format:
    <use_mcp_tool>
    <server_name>server name</server_name>
    <tool_name>tool name</tool_name>
    <arguments>
    {...}
    </arguments>
    </use_mcp_tool>
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        
        # State tracking for streaming
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []
        self.buffer: str = ""  # Buffer for potential tool call tags
        
        # Token definitions
        self.tool_call_start_token: str = "<use_mcp_tool>"
        self.tool_call_end_token: str = "</use_mcp_tool>"
        
        # Regex patterns
        self.tool_call_regex = re.compile(
            r'<use_mcp_tool>\s*'
            r'<server_name>(.*?)</server_name>\s*'
            r'<tool_name>(.*?)</tool_name>\s*'
            r'<arguments>\s*(.*?)\s*</arguments>\s*'
            r'</use_mcp_tool>',
            re.DOTALL
        )
        
        # For streaming partial tool calls
        # IMPORTANT: Use GREEDY matching (.*) for arguments to capture all content
        # in streaming mode. We'll clean up </arguments> tag in the code if present.
        # The outer ()? makes the whole <arguments> section optional
        # The inner (.*) will match empty string if <arguments> exists but has no content yet
        self.partial_tool_regex = re.compile(
            r'<use_mcp_tool>\s*'
            r'(?:<server_name>(.*?)</server_name>\s*)?'
            r'(?:<tool_name>(.*?)</tool_name>\s*)?'
            r'(?:<arguments>(\s*.*))?',  # Move \s* inside capture group so empty match returns ""
            re.DOTALL
        )

    def _resolve_tool_name(self, server_name: str, tool_name: str, request: ChatCompletionRequest) -> str:
        """
        Resolve the actual tool name by combining server_name and tool_name
        if server_name is not 'default'.
        """
        if not server_name or server_name == "default":
            return tool_name
            
        if not request or not request.tools:
            return tool_name
            
        # Filter tools that contain server_name
        candidates = []
        for tool in request.tools:
            if hasattr(tool, "function") and hasattr(tool.function, "name"):
                name = tool.function.name
                if tool_name in name:
                    candidates.append(name)
        if len(candidates) == 1:
            return candidates[0]
        # Find match containing tool_name
        for candidate in candidates:
            if server_name in candidate:
                logger.debug("Resolved tool %s -> %s (server: %s)", tool_name, candidate, server_name)
                return candidate
                
        return tool_name

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Do not skip special tokens for proper tool parsing
            request.skip_special_tokens = False
        return request
    
    def _ensure_tool_id_valid(self, tool_id: int) -> bool:
        """Ensure the tool_id is valid and arrays have enough elements"""
        if tool_id < 0:
            return False
        
        # Ensure arrays are large enough
        while len(self.streamed_args_for_tool) <= tool_id:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= tool_id:
            self.prev_tool_call_arr.append({})
        
        return True

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Sanity check; avoid unnecessary processing
        logger.debug("model_output: %s", model_output)
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        
        try:
            tool_calls = []
            # Find all complete tool calls
            for match in self.tool_call_regex.finditer(model_output):
                server_name = match.group(1).strip()
                tool_name = match.group(2).strip()
                arguments_str = match.group(3).strip()
                
                # Resolve tool name
                tool_name = self._resolve_tool_name(server_name, tool_name, request)
                
                try:
                    # Parse arguments as JSON
                    arguments = json.loads(arguments_str)
                    
                    tool_call = ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tool_name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                    tool_calls.append(tool_call)
                    
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                    continue
            
            # Extract content before first tool call
            content = model_output[:model_output.find(self.tool_call_start_token)]
            
            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content else None,
            )
            
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)
        
        # Reset state if this is the start of a new request
        if not previous_text:
            self.current_tool_name_sent = False
            self.prev_tool_call_arr = []
            self.current_tool_id = -1
            self.streamed_args_for_tool = []
            self.buffer = ""
            logger.debug("Reset streaming state for new request")
        
        # Check if current text contains complete tool call tag
        # If not, we need to check if we're building up to one
        if self.tool_call_start_token not in current_text:
            # Check if current_text ends with a prefix of tool_call_start_token
            # This handles cases where the tag is being streamed character by character
            
            # Check if current_text (not buffer) could end with a tag prefix
            is_potential_tag = False
            for i in range(1, min(len(self.tool_call_start_token), len(current_text) + 1)):
                suffix = current_text[-i:]
                if self.tool_call_start_token.startswith(suffix):
                    is_potential_tag = True
                    logger.debug("Detected potential tag prefix: '%s'", suffix)
                    break
            
            if is_potential_tag:
                # Calculate what content can be safely sent
                # Only send content that definitely won't be part of the tag
                safe_content_end = len(current_text) - len(suffix)
                safe_content = current_text[len(previous_text):safe_content_end]
                
                if safe_content:
                    logger.debug("Sending safe content before potential tag: %s", safe_content)
                    # Buffer the potential tag prefix for next iteration
                    self.buffer = suffix
                    return DeltaMessage(content=safe_content)
                else:
                    # Nothing safe to send, buffer everything
                    logger.debug("Buffering entire delta as potential tag prefix")
                    self.buffer += delta_text
                    return None
            else:
                # Not a tag prefix, send everything including buffer
                if self.buffer:
                    logger.debug("Flushing buffer with delta: %s", self.buffer + delta_text)
                    result = self.buffer + delta_text
                    self.buffer = ""
                    return DeltaMessage(content=result)
                else:
                    logger.debug("No tool call tokens found!")
                    return DeltaMessage(content=delta_text)
        
        try:
            # Clear buffer once we have a complete tag
            if self.buffer and self.tool_call_start_token in current_text:
                logger.debug("Tool call tag detected, clearing buffer")
                self.buffer = ""
            
            # Figure out where we are in parsing by counting tool call tags
            prev_tool_start_count = previous_text.count(self.tool_call_start_token)
            prev_tool_end_count = previous_text.count(self.tool_call_end_token)
            cur_tool_start_count = current_text.count(self.tool_call_start_token)
            cur_tool_end_count = current_text.count(self.tool_call_end_token)
            
            # Case: starting a new tool call
            # Moved up to ensure we update state before any early returns (e.g. tag suppression)
            if (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count > prev_tool_start_count
            ):
                # Only increment if we're not already tracking this tool
                if self.current_tool_id < cur_tool_start_count - 1:
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    self.streamed_args_for_tool.append("")
                    self.prev_tool_call_arr.append({})
                    logger.debug("Starting new tool %s", self.current_tool_id)
            
            # Case: tool call tag just became complete in current_text
            # This means the tag was spread across multiple deltas
            if (
                prev_tool_start_count < cur_tool_start_count
                and self.tool_call_start_token not in previous_text
            ):
                # Find where the complete tag appears in current_text
                tag_start_idx = current_text.find(self.tool_call_start_token)
                
                # Send any content that was before the tag started
                if tag_start_idx > 0:
                    # Need to figure out what content before the tag hasn't been sent yet
                    content_before_tag = current_text[:tag_start_idx]
                    unsent_content = content_before_tag[len(previous_text):]
                    if unsent_content:
                        logger.debug("Sending content before tool call: %s", unsent_content)
                        return DeltaMessage(content=unsent_content)
                
                # Tag started, suppress it from output
                logger.debug("Tool call started, suppressing tag from output")
                return None
            
            # Case: delta contains the start of a tool call (tag was complete in delta)
            if (
                self.tool_call_start_token in delta_text
                and prev_tool_start_count < cur_tool_start_count
            ):
                # Extract content before the tool call tag in delta_text
                content_before_tag = delta_text.split(self.tool_call_start_token)[0]
                if content_before_tag:
                    logger.debug("Sending content before tool call (from delta): %s", content_before_tag)
                    return DeltaMessage(content=content_before_tag)
                else:
                    # No content before tag, return None to suppress the tag
                    logger.debug("Tool call started in delta, suppressing tag from output")
                    return None
            
            # Case: generating text content (not in tool call)
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
                and self.tool_call_start_token not in delta_text
            ):
                logger.debug("Generating text content! skipping tool parsing.")
                return DeltaMessage(content=delta_text)
            
            # Case: tool call just ended
            content_after_tool_end = None
            tool_call_just_ended = False
            
            if (
                cur_tool_end_count > prev_tool_end_count
                and self.tool_call_end_token in delta_text
            ):
                tool_call_just_ended = True
                # Extract content after the closing tag if any
                content_after_tag = delta_text.split(self.tool_call_end_token)[-1]
                self.buffer = ""
                if content_after_tag:
                    logger.debug("Content detected after tool call: %s", content_after_tag)
                    content_after_tool_end = content_after_tag
            
            # Extract current tool call portion
            if cur_tool_start_count > cur_tool_end_count or tool_call_just_ended:
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
            else:
                return None
            
            # Try to parse current tool call
            try:
                match = self.partial_tool_regex.search(
                    self.tool_call_start_token + tool_call_portion
                )
                if not match:
                    if content_after_tool_end:
                        return DeltaMessage(content=content_after_tool_end)
                    return None
                
                server_name = match.group(1)
                tool_name = match.group(2)
                arguments_str = match.group(3)
                
                if server_name:
                    server_name = server_name.strip()
                
                if tool_name:
                    tool_name = tool_name.strip()
                    # Try to resolve tool name if we have both server and tool name
                    if server_name:
                        tool_name = self._resolve_tool_name(server_name, tool_name, request)
                
                # Clean up arguments_str: remove closing tags if present
                # Since we use greedy matching, we may capture </arguments> and </use_mcp_tool>
                if arguments_str:
                    # Remove </arguments> and everything after it
                    if '</arguments>' in arguments_str:
                        arguments_str = arguments_str.split('</arguments>')[0]
                    elif '</use_mcp_tool>' in arguments_str:
                        arguments_str = arguments_str.split('</use_mcp_tool>')[0]
                    else:
                        # Check for partial tags at the end to avoid sending them
                        # This handles cases like "...</arg" where the tag is split across chunks
                        for tag in ['</arguments>', '</use_mcp_tool>']:
                            for i in range(len(tag) - 1, 0, -1):
                                suffix = tag[:i]
                                if arguments_str.endswith(suffix):
                                    arguments_str = arguments_str[:-len(suffix)]
                                    break
                
                logger.debug("Extracted from regex - server_name: %s, tool_name: %s, arguments_str: %s", 
                            server_name, tool_name, repr(arguments_str))
                
                # Build current tool call dict
                current_tool_call = {}
                if tool_name:
                    current_tool_call["name"] = tool_name.strip()
                if arguments_str:
                    # Store the raw arguments string for streaming
                    # We don't need to parse it with partial_json_parser for streaming
                    # Just mark that we have arguments
                    current_tool_call["arguments_str"] = arguments_str
                    # Try to parse for validation but don't block streaming if it fails
                    flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR
                    try:
                        args = partial_json_parser.loads(arguments_str.strip(), flags)
                        current_tool_call["arguments"] = args
                        logger.debug("Successfully parsed arguments: %s", args)
                    except Exception as e:
                        logger.debug("Cannot parse arguments yet: %s", e)
                
            except Exception:
                logger.debug("Error parsing partial tool call")
                if content_after_tool_end:
                    return DeltaMessage(content=content_after_tool_end)
                return None
            
            # Case: haven't sent tool name yet
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")
                if function_name:
                    self.current_tool_name_sent = True
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=make_tool_call_id(),
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                else:
                    return None
            
            # Case: handle arguments delta
            # Ensure tool_id is valid and arrays are properly sized
            if not self._ensure_tool_id_valid(self.current_tool_id):
                logger.debug("Invalid tool_id: %s", self.current_tool_id)
                return None
            
            # Use arguments_str instead of parsed arguments to decide if we have arguments
            prev_arguments_str = self.prev_tool_call_arr[self.current_tool_id].get("arguments_str", "")
            cur_arguments_str = current_tool_call.get("arguments_str", "")
            
            logger.debug("diffing old arguments_str: %s", repr(prev_arguments_str))
            logger.debug("against new arguments_str: %s", repr(cur_arguments_str))
            
            # No arguments yet
            if not cur_arguments_str and not prev_arguments_str:
                logger.debug("Skipping - no arguments yet")
                if content_after_tool_end:
                    return DeltaMessage(content=content_after_tool_end)
                return None
            
            # First arguments available
            elif cur_arguments_str and not prev_arguments_str:
                # Send the first batch of arguments
                arguments_delta = cur_arguments_str.strip()
                logger.debug("First arguments received: %s", repr(arguments_delta))
                
                # Arrays are already ensured by _ensure_tool_id_valid above
                self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
                
                logger.debug("Sending first arguments delta to client")
                return DeltaMessage(
                    content=content_after_tool_end,
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=arguments_delta
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
            
            # Update to existing arguments - stream the new text
            elif cur_arguments_str and prev_arguments_str:
                # Calculate what's new in this delta
                if len(cur_arguments_str) > len(prev_arguments_str):
                    # New content added
                    new_content = cur_arguments_str[len(prev_arguments_str):]
                    logger.debug("Arguments delta (incremental): %s", repr(new_content))
                    
                    # Arrays are already ensured by _ensure_tool_id_valid above
                    self.streamed_args_for_tool[self.current_tool_id] += new_content
                    self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
                    
                    return DeltaMessage(
                        content=content_after_tool_end,
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=new_content
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
            
            if content_after_tool_end:
                return DeltaMessage(content=content_after_tool_end)
            
            return None
            
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None


# Register the tool parser to ToolParserManager
ToolParserManager.register_module(
    "mirothinker", True ,MirothinkerToolParser
)
