# =============================================================================
# CLI CHAT — learning-mcp
# =============================================================================
# CliChat extends the base Chat class with document-specific features.
# It demonstrates how a client application bridges MCP primitives into a
# coherent user experience:
#
#   • Resources  → pre-populate context before calling the model (@mentions)
#   • Prompts    → slash commands that inject server-crafted messages (/summarize)
#   • Tools      → handled generically by the base Chat agentic loop
# =============================================================================

from typing import List, Tuple
from mcp.types import Prompt, PromptMessage
from anthropic.types import MessageParam

from core.chat import Chat
from core.claude import Claude
from mcp_client import MCPClient


class CliChat(Chat):
    """
    Document-aware chat session for the CLI application.

    Extends Chat with:
      - @mention syntax  : type "@report.pdf" to inject the doc into context
      - /command syntax  : type "/summarize plan.md" to trigger an MCP prompt
    """

    def __init__(
        self,
        doc_client: MCPClient,
        clients: dict[str, MCPClient],
        claude_service: Claude,
    ):
        # Pass all clients to the base class so the agentic loop can call
        # tools from any of them.
        super().__init__(clients=clients, claude_service=claude_service)

        # doc_client is the MCPClient connected to our DocumentMCP server.
        # We keep a separate reference so we can call document-specific
        # resources and prompts directly.
        self.doc_client: MCPClient = doc_client

    # -------------------------------------------------------------------------
    # Prompt helpers
    # -------------------------------------------------------------------------

    async def list_prompts(self) -> list[Prompt]:
        """
        Return the list of prompts the DocumentMCP server exposes.

        The CLI app calls this at startup to show the user what /commands are
        available (e.g. /summarize, /rewrite_as_markdown).
        """
        return await self.doc_client.list_prompts()

    async def get_prompt(
        self, command: str, doc_id: str
    ) -> list[PromptMessage]:
        """
        Fetch a rendered prompt from the server.

        The server fills in {doc_id} and returns a list of PromptMessage
        objects ready to append to the conversation.

        We unwrap `.messages` from GetPromptResult so callers get the list
        they actually need.
        """
        result = await self.doc_client.get_prompt(command, {"doc_id": doc_id})
        # GetPromptResult.messages is the list of PromptMessage objects.
        return result.messages

    # -------------------------------------------------------------------------
    # Resource helpers
    # -------------------------------------------------------------------------

    async def list_docs_ids(self) -> list[str]:
        """
        Fetch the list of available document IDs from the server resource.

        This uses an MCP Resource (not a Tool) — the distinction matters:
          • Resource → client pulls data directly (no LLM involved)
          • Tool     → LLM decides to call it during inference

        The resource returns a newline-separated string; we split it here.
        """
        contents = await self.doc_client.read_resource("docs://documents")
        # contents is a list of TextContent/BlobResourceContents items.
        # The docs://documents resource returns a single TextContent with
        # newline-separated IDs.
        if not contents:
            return []
        text = contents[0].text  # TextContent.text is the string payload
        return [line.strip() for line in text.splitlines() if line.strip()]

    async def get_doc_content(self, doc_id: str) -> str:
        """
        Fetch the content of a specific document via the templated resource.

        URI template on server: "docs://documents/{doc_id}"
        We expand it here with the actual doc_id.
        """
        contents = await self.doc_client.read_resource(f"docs://documents/{doc_id}")
        if not contents:
            return ""
        return contents[0].text

    # -------------------------------------------------------------------------
    # Query processing
    # -------------------------------------------------------------------------

    async def _extract_resources(self, query: str) -> str:
        """
        Scan the query for @mentions and fetch matching document contents.

        Example: "Can you compare @report.pdf and @financials.docx?"
        → fetches both docs and wraps them in <document> XML tags so Claude
          can read them as part of the system context.

        This is a client-side RAG (Retrieval Augmented Generation) pattern:
        we enrich the prompt with relevant data before the model call.
        """
        # Extract any words starting with "@" (strip the "@" prefix).
        mentions = [word[1:] for word in query.split() if word.startswith("@")]

        if not mentions:
            return ""

        # Only fetch docs that are actually @mentioned AND exist on the server.
        doc_ids = await self.list_docs_ids()
        mentioned_docs: list[Tuple[str, str]] = []

        for doc_id in doc_ids:
            if doc_id in mentions:
                content = await self.get_doc_content(doc_id)
                mentioned_docs.append((doc_id, content))

        # Wrap each document in XML tags so the model can clearly identify
        # where each document begins and ends.
        return "".join(
            f'\n<document id="{doc_id}">\n{content}\n</document>\n'
            for doc_id, content in mentioned_docs
        )

    async def _process_command(self, query: str) -> bool:
        """
        Handle slash commands by fetching the matching MCP prompt.

        Format: "/<prompt_name> <doc_id>"
        Example: "/summarize plan.md"

        Returns True if the query was a command (so the caller can skip the
        normal query processing path).

        How MCP Prompts work here:
        1. The client sends a `prompts/get` request with the prompt name + args.
        2. The server renders the template and returns PromptMessage objects.
        3. We append those messages to the conversation — the next model call
           sees them as if the user had typed them manually.
        """
        if not query.startswith("/"):
            return False

        parts = query.split()
        if len(parts) < 2:
            print("Usage: /<command> <doc_id>  e.g. /summarize plan.md")
            return True

        command = parts[0].lstrip("/")   # e.g. "summarize"
        doc_id  = parts[1]              # e.g. "plan.md"

        try:
            prompt_messages = await self.get_prompt(command, doc_id)
        except Exception as e:
            print(f"Prompt error: {e}")
            return True

        # Convert server PromptMessages → Anthropic MessageParams and add them
        # to the ongoing conversation. The next model call will "see" these.
        self.messages += convert_prompt_messages_to_message_params(prompt_messages)
        return True

    async def _process_query(self, query: str):
        """
        Entry point for each user message.

        Decision tree:
          1. Slash command?  → fetch MCP prompt, inject, return early.
          2. Has @mentions?  → fetch resource contents, inject into prompt.
          3. Normal query?  → pass straight through to Claude with context.

        The base Chat.run() calls this before every model call.
        """
        # Handle /command syntax first (short-circuits normal flow).
        if await self._process_command(query):
            return

        # Fetch any @mentioned document content via MCP resources.
        added_resources = await self._extract_resources(query)

        # Build a structured prompt that keeps the document context clearly
        # separated from the user's question.
        prompt = f"""
        The user has a question:
        <query>
        {query}
        </query>

        The following context may be useful in answering their question:
        <context>
        {added_resources}
        </context>

        Note: the user's query might contain references like "@report.pdf".
        The "@" is only an @mention marker — the actual filename is "report.pdf".
        If document content is included above, you don't need to call a tool to
        read it again. Answer directly and concisely without mentioning the context.
        """

        self.messages.append({"role": "user", "content": prompt})


# =============================================================================
# HELPER FUNCTIONS — PromptMessage → MessageParam conversion
# =============================================================================
# MCP uses its own `PromptMessage` type. The Anthropic SDK uses `MessageParam`.
# These helpers translate between the two so we can inject MCP prompt responses
# into an Anthropic chat conversation.
# =============================================================================

def convert_prompt_message_to_message_param(
    prompt_message: "PromptMessage",
) -> MessageParam:
    """Convert a single MCP PromptMessage to an Anthropic MessageParam."""
    role = "user" if prompt_message.role == "user" else "assistant"
    content = prompt_message.content

    # content can be a TextContent object or a raw dict (depends on MCP SDK version).
    if isinstance(content, dict) or hasattr(content, "__dict__"):
        content_type = (
            content.get("type") if isinstance(content, dict) else getattr(content, "type", None)
        )
        if content_type == "text":
            text = (
                content.get("text", "") if isinstance(content, dict) else getattr(content, "text", "")
            )
            return {"role": role, "content": text}

    if isinstance(content, list):
        text_blocks = []
        for item in content:
            item_type = (
                item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            )
            if item_type == "text":
                item_text = (
                    item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                )
                text_blocks.append({"type": "text", "text": item_text})
        if text_blocks:
            return {"role": role, "content": text_blocks}

    return {"role": role, "content": ""}


def convert_prompt_messages_to_message_params(
    prompt_messages: List[PromptMessage],
) -> List[MessageParam]:
    """Convert a list of MCP PromptMessages to Anthropic MessageParams."""
    return [convert_prompt_message_to_message_param(msg) for msg in prompt_messages]
