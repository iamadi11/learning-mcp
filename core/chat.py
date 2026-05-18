# =============================================================================
# CHAT — learning-mcp
# =============================================================================
# Base class for the agentic chat loop.
#
# The "agentic loop" is the core pattern when using LLMs with tools:
#
#   ┌─────────────────────────────────────────────────────────────┐
#   │  User message                                               │
#   │       ↓                                                     │
#   │  Claude (with tools list) ──→ tool_use response            │
#   │       ↓                                                     │
#   │  Execute tool via MCP ──→ tool result                      │
#   │       ↓                                                     │
#   │  Feed result back as user message ──→ repeat               │
#   │       ↓ (when stop_reason == "end_turn")                   │
#   │  Final text response                                        │
#   └─────────────────────────────────────────────────────────────┘
#
# The loop runs until Claude stops asking for tools (stop_reason = "end_turn").
# Multiple tool calls can happen in a single turn (Claude may batch them).
# =============================================================================

from core.claude import Claude
from mcp_client import MCPClient
from core.tools import ToolManager
from anthropic.types import MessageParam


class Chat:
    """
    Base agentic chat loop. Subclasses override `_process_query` to
    customise how user input is preprocessed before hitting the model.
    """

    def __init__(self, claude_service: Claude, clients: dict[str, MCPClient]):
        self.claude_service: Claude = claude_service
        # `clients` maps a logical name to an MCPClient. Multiple servers can
        # be connected at once; ToolManager fans out tool discovery across all.
        self.clients: dict[str, MCPClient] = clients
        # Conversation history — Anthropic requires the full message list on
        # every API call. We accumulate it here across turns.
        self.messages: list[MessageParam] = []

    async def _process_query(self, query: str):
        """
        Default pre-processing: just add the raw user message to history.
        CliChat overrides this to handle @mentions and /commands.
        """
        self.messages.append({"role": "user", "content": query})

    async def run(self, query: str) -> str:
        """
        Process one user query through the full agentic loop.

        Returns the final plain-text response from Claude after all tool
        calls (if any) have been resolved.
        """
        final_text_response = ""

        # Let the subclass customise how the query enters the message list.
        await self._process_query(query)

        # Agentic loop — keeps going as long as Claude wants to call tools.
        while True:
            # Gather every tool available across all connected MCP servers
            # and pass them to Claude so it knows what actions are possible.
            all_tools = await ToolManager.get_all_tools(self.clients)

            response = self.claude_service.chat(
                messages=self.messages,
                tools=all_tools,
            )

            # Add Claude's response (which may include tool_use blocks) to
            # history so the next call has full context.
            self.claude_service.add_assistant_message(self.messages, response)

            if response.stop_reason == "tool_use":
                # Claude wants to call one or more tools.
                # Print any text it produced alongside the tool call (e.g.
                # "Let me check that document for you...").
                print(self.claude_service.text_from_message(response))

                # Execute every tool_use block in the response via the matching
                # MCP server, then wrap the results as a user message.
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )

                # Tool results are sent back as a "user" role message in the
                # Anthropic API — this is how the model "sees" what the tool
                # returned before deciding what to do next.
                self.claude_service.add_user_message(
                    self.messages, tool_result_parts
                )

            else:
                # stop_reason == "end_turn" — Claude is done using tools and
                # has produced its final answer. Extract the text and exit.
                final_text_response = self.claude_service.text_from_message(response)
                break

        return final_text_response
