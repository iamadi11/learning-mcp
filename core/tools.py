# =============================================================================
# TOOL MANAGER — learning-mcp
# =============================================================================
# ToolManager handles everything related to MCP tools on the *client* side:
#
#   1. Discovery — ask every connected server what tools it exposes, then
#      reformat them as Anthropic tool definitions so Claude can see them.
#
#   2. Routing   — when Claude asks to call a tool, figure out which MCP
#      server owns that tool and forward the call there.
#
#   3. Result mapping — convert the MCP CallToolResult back into the
#      ToolResultBlockParam format that the Anthropic API expects.
#
# Having this in a separate class keeps it reusable across Chat subclasses.
# =============================================================================

import json
from typing import Optional, Literal, List
from mcp.types import CallToolResult, Tool, TextContent
from mcp_client import MCPClient
from anthropic.types import Message, ToolResultBlockParam


class ToolManager:

    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict]:
        """
        Collect tools from every connected MCP server and reformat them for
        the Anthropic API.

        MCP tool schema  →  Anthropic tool definition:
          tool.name        →  "name"
          tool.description →  "description"
          tool.inputSchema →  "input_schema"  (already a JSON Schema dict)

        Claude receives this list on every API call so it knows what actions
        are available regardless of which server they live on.
        """
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            tools += [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tool_models
            ]
        return tools

    @classmethod
    async def _find_client_with_tool(
        cls, clients: list[MCPClient], tool_name: str
    ) -> Optional[MCPClient]:
        """
        Search connected servers to find the one that owns a given tool.

        This is O(n_clients × n_tools) — fine for learning, but in production
        you'd cache the tool→client mapping at startup to avoid repeat lookups.
        """
        for client in clients:
            tools = await client.list_tools()
            if any(t.name == tool_name for t in tools):
                return client
        return None

    @classmethod
    def _build_tool_result_part(
        cls,
        tool_use_id: str,
        text: str,
        status: Literal["success", "error"],
    ) -> ToolResultBlockParam:
        """
        Build a ToolResultBlockParam — the format the Anthropic API uses to
        pass tool output back into the conversation.

        tool_use_id links the result to the specific `tool_use` block Claude
        emitted (Claude may call multiple tools in one response; each gets its
        own ID so the results can be matched up).
        """
        return {
            "tool_use_id": tool_use_id,
            "type": "tool_result",
            "content": text,
            "is_error": status == "error",
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], message: Message
    ) -> List[ToolResultBlockParam]:
        """
        Execute every tool_use block in a Claude response.

        Steps for each tool_use block:
          1. Find the MCP client that owns the tool.
          2. Call the tool with the arguments Claude provided.
          3. Extract TextContent items from the result.
          4. Return a ToolResultBlockParam for each call.

        The results list is fed back as a "user" message in the next API call
        so Claude can see what the tools returned and decide what to do next.
        """
        # Filter the response content to only tool_use blocks.
        tool_requests = [block for block in message.content if block.type == "tool_use"]
        tool_result_blocks: list[ToolResultBlockParam] = []

        for tool_request in tool_requests:
            tool_use_id = tool_request.id      # unique per tool call in this turn
            tool_name   = tool_request.name    # which tool Claude wants to call
            tool_input  = tool_request.input   # arguments as a dict

            # Route: find which server owns this tool.
            client = await cls._find_client_with_tool(
                list(clients.values()), tool_name
            )

            if not client:
                # Tool not found on any connected server — return an error
                # result so Claude can acknowledge and recover gracefully.
                tool_result_blocks.append(
                    cls._build_tool_result_part(tool_use_id, "Tool not found on any connected server.", "error")
                )
                continue

            tool_output: CallToolResult | None = None
            try:
                # Execute the tool call via the MCP client.
                tool_output = await client.call_tool(tool_name, tool_input)

                # MCP tool results are lists of content blocks. We extract
                # just the text items here. ImageContent would need separate
                # handling for multimodal use cases.
                text_items = [
                    item.text for item in (tool_output.content or [])
                    if isinstance(item, TextContent)
                ]
                content_json = json.dumps(text_items)

                tool_result_blocks.append(
                    cls._build_tool_result_part(
                        tool_use_id,
                        content_json,
                        "error" if (tool_output and tool_output.isError) else "success",
                    )
                )

            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_blocks.append(
                    cls._build_tool_result_part(
                        tool_use_id,
                        json.dumps({"error": error_message}),
                        "error",
                    )
                )

        return tool_result_blocks
