# =============================================================================
# MCP CLIENT — learning-mcp
# =============================================================================
# The MCP client is the *consumer* side of the Model Context Protocol.
# It connects to an MCP server and exposes a clean Python API for:
#
#   • Discovering and calling Tools
#   • Fetching Resources
#   • Fetching Prompts
#
# Transport layer (how client and server communicate):
# -------------------------------------------------------
# MCP supports multiple transports. This client uses **stdio**, which means:
#   1. The client launches the server as a child process (subprocess).
#   2. Communication happens over the process's stdin/stdout pipes.
#   3. Messages are JSON-RPC 2.0 frames — FastMCP and this SDK handle all
#      the serialisation/deserialisation for you.
#
# The `mcp` Python package provides:
#   • ClientSession  — manages the MCP protocol session (init, request, notify)
#   • stdio_client() — async context manager that spawns the subprocess and
#                      returns (read_stream, write_stream) async streams
# =============================================================================

import sys
import asyncio
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


class MCPClient:
    """
    A thin, reusable wrapper around `ClientSession` for a single MCP server.

    Lifecycle
    ---------
    1. Construct with the command + args needed to launch the server process.
    2. Call `connect()` (or use as an async context manager) to spawn the
       server, run the MCP initialisation handshake, and be ready to use.
    3. Use `list_tools()`, `call_tool()`, `list_resources()`, `read_resource()`,
       `list_prompts()`, `get_prompt()` to interact with the server.
    4. Call `cleanup()` (or exit the context manager) to shut everything down.

    The AsyncExitStack makes it easy to manage multiple async context managers
    (the transport and the session) and clean them all up in reverse order.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        env: Optional[dict] = None,
    ):
        # The command + args are what you'd type in a terminal to start the
        # server, e.g. ("uv", ["run", "mcp_server.py"]) or ("python", ["server.py"]).
        self._command = command
        self._args = args
        # Optional extra environment variables forwarded to the server process.
        self._env = env

        self._session: Optional[ClientSession] = None
        # AsyncExitStack lets us register multiple async context managers and
        # ensures they are all cleaned up (in LIFO order) on exit.
        self._exit_stack = AsyncExitStack()

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    async def connect(self):
        """
        Spawn the MCP server subprocess and run the protocol handshake.

        Step-by-step:
        1. StdioServerParameters bundles the launch config (command, args, env).
        2. stdio_client() spawns the subprocess and returns two async streams:
             read_stream  — data arriving FROM the server (server → client)
             write_stream — data going   TO   the server (client → server)
        3. ClientSession wraps those streams and implements the MCP protocol:
             - It sends an `initialize` request to negotiate capabilities.
             - After that, it's ready to send tool/resource/prompt requests.
        """

        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )

        # Enter the stdio_client context — this launches the server subprocess.
        # Registering it with the exit_stack means it will be cleaned up (i.e.
        # the subprocess will be terminated) when we call cleanup().
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        read_stream, write_stream = stdio_transport

        # Create and enter a ClientSession over those streams.
        # ClientSession handles all JSON-RPC framing and protocol state.
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # initialize() sends the MCP `initialize` request.
        # The server replies with its name, version, and capability list
        # (which primitives it supports: tools, resources, prompts, etc.).
        await self._session.initialize()

    @property
    def session(self) -> ClientSession:
        """Guard: raises a clear error if you forget to call connect() first."""
        if self._session is None:
            raise ConnectionError(
                "Client session not initialized. Call connect() first."
            )
        return self._session

    # -------------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------------
    # Tools are functions the LLM can call to take actions or fetch data.
    # list_tools() → the client discovers what's available (usually at startup).
    # call_tool()  → the client executes a specific tool with arguments.
    # -------------------------------------------------------------------------

    async def list_tools(self) -> list[types.Tool]:
        """
        Fetch all tools the server exposes.

        Each `Tool` object has:
          • name        — unique identifier used when calling the tool
          • description — plain-English description shown to the LLM
          • inputSchema — JSON Schema describing the expected arguments

        The client typically converts this list into the `tools` parameter
        for an Anthropic `messages.create()` call so Claude knows what it can do.
        """
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(
        self,
        tool_name: str,
        tool_input: dict,
    ) -> types.CallToolResult:
        """
        Invoke a named tool with the given arguments and return the result.

        `CallToolResult` has:
          • content  — list of content blocks (TextContent, ImageContent, etc.)
          • isError  — True if the tool raised an exception on the server

        The agentic loop in core/chat.py calls this after Claude responds with
        a `tool_use` block, then feeds the result back into the next message.
        """
        return await self.session.call_tool(tool_name, tool_input)

    # -------------------------------------------------------------------------
    # Resources
    # -------------------------------------------------------------------------
    # Resources are read-only data blobs addressed by URI (like a REST GET).
    # They are NOT automatically surfaced to the LLM — the client decides when
    # to fetch them (e.g. to pre-populate context before calling the model).
    # -------------------------------------------------------------------------

    async def list_resources(self) -> list[types.Resource]:
        """
        Fetch metadata for all resources the server exposes.

        Each `Resource` has:
          • uri         — the address used to read the resource
          • name        — human-readable label
          • description — what the resource contains
          • mimeType    — optional content type hint (e.g. "text/plain")

        Note: list_resources only returns *static* URIs. Templated resources
        (like "docs://documents/{doc_id}") show up in list_resource_templates().
        """
        result = await self.session.list_resources()
        return result.resources

    async def list_resource_templates(self) -> list[types.ResourceTemplate]:
        """
        Fetch URI templates for parameterised resources.

        A ResourceTemplate has a `uriTemplate` field (RFC 6570 URI template),
        e.g. "docs://documents/{doc_id}". The client fills in the variables
        and calls read_resource() with the expanded URI.
        """
        result = await self.session.list_resource_templates()
        return result.resourceTemplates

    async def read_resource(self, uri: str) -> Any:
        """
        Fetch the content of a resource by its URI.

        Returns `result.contents` — a list of content items (usually one).
        TextContent items have a `.text` attribute; BlobResourceContents
        items have `.blob` (base64 bytes).

        Examples:
          await client.read_resource("docs://documents")
          await client.read_resource("docs://documents/report.pdf")
        """
        result = await self.session.read_resource(uri)
        return result.contents

    # -------------------------------------------------------------------------
    # Prompts
    # -------------------------------------------------------------------------
    # Prompts are reusable message templates stored on the server.
    # The client fetches them, gets back rendered PromptMessage objects, and
    # injects them into the conversation — no LLM call needed to generate them.
    # -------------------------------------------------------------------------

    async def list_prompts(self) -> list[types.Prompt]:
        """
        Fetch metadata for all prompts the server exposes.

        Each `Prompt` has:
          • name        — identifier used to fetch the prompt
          • description — what the prompt does
          • arguments   — list of PromptArgument (name, description, required)
        """
        result = await self.session.list_prompts()
        return result.prompts

    async def get_prompt(
        self,
        prompt_name: str,
        args: dict[str, str] | None = None,
    ) -> types.GetPromptResult:
        """
        Fetch a rendered prompt from the server.

        The server fills in any template variables (from `args`) and returns
        a list of PromptMessage objects ready to append to the conversation.

        `GetPromptResult` has:
          • description — optional description of this specific rendering
          • messages    — list[PromptMessage], each with role + content

        Example:
          result = await client.get_prompt("summarize", {"doc_id": "plan.md"})
          # result.messages is a list of PromptMessage ready for the chat loop
        """
        result = await self.session.get_prompt(
            prompt_name,
            arguments=args or {},
        )
        return result

    # -------------------------------------------------------------------------
    # Cleanup & context manager support
    # -------------------------------------------------------------------------

    async def cleanup(self):
        """
        Close the session and terminate the server subprocess.

        AsyncExitStack.aclose() runs all registered cleanup callbacks in
        reverse order — first the session, then the transport (subprocess).
        """
        await self._exit_stack.aclose()
        self._session = None

    async def __aenter__(self):
        """Support `async with MCPClient(...) as client:` usage."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()


# =============================================================================
# QUICK TEST / DEMO
# =============================================================================
# Run this file directly to see all MCP primitives printed to the console:
#   uv run mcp_client.py
# =============================================================================

async def main():
    """
    Connects to the local MCP server and prints every primitive it exposes
    (tools, resources, resource templates, prompts), then exercises a few of them.
    """
    import sys as _sys
    # Use the same Python interpreter that's running this script so the
    # server process shares the same virtual environment and packages.
    async with MCPClient(
        command=_sys.executable,
        args=["mcp_server.py"],
    ) as client:

        # --- Tools -----------------------------------------------------------
        print("\n=== TOOLS ===")
        tools = await client.list_tools()
        for tool in tools:
            print(f"  • {tool.name}")
            print(f"    {tool.description}")

        # --- Resources (static) ----------------------------------------------
        print("\n=== RESOURCES ===")
        resources = await client.list_resources()
        for r in resources:
            print(f"  • {r.uri}  ({r.name})")
            print(f"    {r.description}")

        # --- Resource templates (parameterised) ------------------------------
        print("\n=== RESOURCE TEMPLATES ===")
        templates = await client.list_resource_templates()
        for t in templates:
            print(f"  • {t.uriTemplate}  ({t.name})")

        # --- Prompts ---------------------------------------------------------
        print("\n=== PROMPTS ===")
        prompts = await client.list_prompts()
        for p in prompts:
            args_desc = ", ".join(a.name for a in (p.arguments or []))
            print(f"  • {p.name}({args_desc})")
            print(f"    {p.description}")

        # --- Exercise: read the docs:// resource -----------------------------
        print("\n=== READ RESOURCE: docs://documents ===")
        contents = await client.read_resource("docs://documents")
        for item in contents:
            print(item.text)

        # --- Exercise: call list_documents tool ------------------------------
        print("\n=== CALL TOOL: list_documents ===")
        result = await client.call_tool("list_documents", {})
        for item in result.content:
            print(item.text)

        # --- Exercise: call search_documents tool ----------------------------
        print("\n=== CALL TOOL: search_documents(keyword='project') ===")
        result = await client.call_tool("search_documents", {"keyword": "project"})
        for item in result.content:
            print(item.text)

        # --- Exercise: fetch the summarize prompt ----------------------------
        print("\n=== GET PROMPT: summarize(doc_id='plan.md') ===")
        prompt_result = await client.get_prompt("summarize", {"doc_id": "plan.md"})
        for msg in prompt_result.messages:
            print(f"  [{msg.role}]: {msg.content.text[:120]}...")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
