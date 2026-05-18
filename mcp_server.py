# =============================================================================
# MCP SERVER — learning-mcp
# =============================================================================
# An MCP server exposes three kinds of primitives to any connected client:
#
#   • Tools     — functions Claude (or any LLM) can CALL to take actions or
#                 retrieve data (think: API calls, DB writes, calculations).
#
#   • Resources — read-only data the client can pull on demand, identified by
#                 a URI (think: files, DB rows, live sensor readings).
#                 Unlike tools, the server doesn't push resources automatically;
#                 the client asks for them when it needs them.
#
#   • Prompts   — reusable prompt *templates* the client can fetch and inject
#                 into a conversation. Great for standardising how you ask the
#                 model to do common tasks (summarise, reformat, etc.).
#
# FastMCP (from the `mcp` package) lets you define all three using simple
# Python decorators, so you don't have to write any JSON-RPC boilerplate.
# =============================================================================

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, PromptMessage
from pydantic import Field

# --- Server instantiation ----------------------------------------------------
# FastMCP wraps the MCP protocol for you. The name ("DocumentMCP") is sent to
# the client during the handshake so it can identify which server it's talking
# to. log_level="ERROR" keeps the console quiet while learning.
mcp = FastMCP("DocumentMCP", log_level="ERROR")


# =============================================================================
# DATA STORE
# =============================================================================
# For learning purposes we use a plain dict as an in-memory document store.
# In a real server this would be a DB query, S3 call, filesystem read, etc.
# The key is the document ID; the value is its content.
# =============================================================================

docs: dict[str, str] = {
    "deposition.md":  "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf":     "The report details the state of a 20m condenser tower.",
    "financials.docx":"These financials outline the project's budget and expenditures.",
    "outlook.pdf":    "This document presents the projected future performance of the system.",
    "plan.md":        "The plan outlines the steps for the project's implementation.",
    "spec.txt":       "These specifications define the technical requirements for the equipment.",
}


# =============================================================================
# TOOLS
# =============================================================================
# Tools are the primary way an LLM takes *actions* through an MCP server.
# The LLM sees the name + description and decides when to call the tool.
# Parameters are described via type annotations and Pydantic Field metadata —
# FastMCP turns these into a JSON Schema that the client sends to the model.
# =============================================================================

@mcp.tool(
    name="read_doc_contents",
    description="Read the content of a document and return it as a string.",
)
def read_document(
    doc_id: str = Field(description="ID of the document to read (e.g. 'report.pdf')"),
) -> str:
    """
    Tool: read_doc_contents
    -----------------------
    The LLM will call this when it needs to inspect a document's content.
    Tools can raise exceptions — FastMCP catches them and returns an error
    result to the client instead of crashing the server.
    """
    if doc_id not in docs:
        # Raising ValueError surfaces a user-friendly error inside the chat.
        raise ValueError(f"Document '{doc_id}' not found. Available: {list(docs.keys())}")
    return docs[doc_id]


@mcp.tool(
    name="edit_document",
    description="Edit a document by replacing an exact substring with new text.",
)
def edit_document(
    doc_id: str = Field(description="ID of the document to edit"),
    old_str: str = Field(description="Exact text to replace (must match including whitespace)"),
    new_str: str = Field(description="Replacement text"),
) -> str:
    """
    Tool: edit_document
    -------------------
    Mutates the in-memory store. In a real server you'd persist to disk/DB.
    Returning a confirmation string lets the LLM know the action succeeded.
    """
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found.")
    if old_str not in docs[doc_id]:
        raise ValueError(f"Text '{old_str}' not found in '{doc_id}'.")

    docs[doc_id] = docs[doc_id].replace(old_str, new_str)
    return f"Document '{doc_id}' updated successfully."


@mcp.tool(
    name="list_documents",
    description="Return a list of all available document IDs.",
)
def list_documents() -> list[str]:
    """
    Tool: list_documents
    --------------------
    Shows the difference between a Tool and a Resource:
    - This *tool* lets the LLM discover available docs mid-conversation.
    - The 'docs://documents' resource below lets the CLIENT pre-load the
      same list (e.g. to populate a UI picker) without involving the LLM.
    Both expose similar data, but through different MCP primitives with
    different access patterns.
    """
    return list(docs.keys())


@mcp.tool(
    name="create_document",
    description="Create a new document with the given ID and initial content.",
)
def create_document(
    doc_id: str = Field(description="Unique ID for the new document (e.g. 'notes.txt')"),
    content: str = Field(description="Initial content for the document"),
) -> str:
    """
    Tool: create_document
    ---------------------
    Demonstrates how tools can write new state, not just read it.
    """
    if doc_id in docs:
        raise ValueError(f"Document '{doc_id}' already exists. Use edit_document to modify it.")
    docs[doc_id] = content
    return f"Document '{doc_id}' created."


@mcp.tool(
    name="delete_document",
    description="Permanently delete a document by its ID.",
)
def delete_document(
    doc_id: str = Field(description="ID of the document to delete"),
) -> str:
    """
    Tool: delete_document
    ---------------------
    Demonstrates destructive tool actions. In production you'd want soft
    deletes or confirmation flows here.
    """
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found.")
    del docs[doc_id]
    return f"Document '{doc_id}' deleted."


@mcp.tool(
    name="search_documents",
    description="Search all documents for a keyword and return matching doc IDs with snippets.",
)
def search_documents(
    keyword: str = Field(description="Case-insensitive keyword to search for"),
) -> list[dict]:
    """
    Tool: search_documents
    ----------------------
    A more complex tool that returns structured data (list of dicts).
    FastMCP serialises the return value to JSON automatically.
    """
    results = []
    for doc_id, content in docs.items():
        if keyword.lower() in content.lower():
            # Find the position and grab a short snippet around the match.
            idx = content.lower().index(keyword.lower())
            start = max(0, idx - 30)
            end = min(len(content), idx + len(keyword) + 30)
            snippet = ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")
            results.append({"doc_id": doc_id, "snippet": snippet})

    if not results:
        return [{"message": f"No documents contain '{keyword}'."}]
    return results


# =============================================================================
# RESOURCES
# =============================================================================
# Resources are *pull-based*, read-only data sources identified by a URI.
# The client requests them explicitly — the LLM is NOT automatically aware
# of resources; it's up to the client to decide when to fetch them.
#
# URI syntax in FastMCP:
#   - Static URI  : @mcp.resource("scheme://path")
#   - Template URI: @mcp.resource("scheme://path/{variable}")
#                   FastMCP extracts {variable} and passes it as a parameter.
#
# The return value is sent back as a resource "contents" list.
# Strings become TextContent; bytes become BlobResourceContents.
# =============================================================================

@mcp.resource(
    uri="docs://documents",
    name="Document List",
    description="Lists all available document IDs as a newline-separated string.",
)
def resource_list_documents() -> str:
    """
    Resource: docs://documents
    --------------------------
    Static resource — same URI always returns the current document list.
    The client in cli_chat.py fetches this to know which @mentions are valid
    before passing the query to Claude.
    """
    # Return newline-separated IDs so the client can split them easily.
    return "\n".join(docs.keys())


@mcp.resource(
    uri="docs://documents/{doc_id}",
    name="Document Content",
    description="Returns the full content of a single document.",
)
def resource_get_document(doc_id: str) -> str:
    """
    Resource: docs://documents/{doc_id}
    ------------------------------------
    Templated resource — FastMCP parses the URI and injects `doc_id`.
    The client uses this to hydrate @mentions in the user's query with
    actual document text before sending the message to Claude.

    Note: this looks similar to the read_doc_contents *tool*, but the
    intended caller is different — a resource is fetched by client-side
    logic, a tool is called by the LLM during inference.
    """
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found.")
    return docs[doc_id]


# =============================================================================
# PROMPTS
# =============================================================================
# Prompts are server-side *message templates* that the client can fetch and
# inject into the conversation. They let you centralise prompt engineering on
# the server so every client (CLI, web app, IDE plugin…) uses the same
# carefully-crafted instructions.
#
# A prompt function returns a list of PromptMessage objects — each has:
#   • role    : "user" or "assistant"
#   • content : a TextContent (or other content block) with the message text
#
# The client fetches the rendered messages and appends them to the ongoing
# conversation before the next model call.
# =============================================================================

@mcp.prompt(
    name="summarize",
    description="Generate a prompt that asks the model to summarize a document.",
)
def prompt_summarize(
    doc_id: str = Field(description="ID of the document to summarize"),
) -> list[PromptMessage]:
    """
    Prompt: summarize
    -----------------
    The server fetches the document content and bakes it into the prompt so
    the model receives a fully self-contained instruction — the client doesn't
    need to know how to retrieve or format the document.
    """
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found.")

    content = docs[doc_id]

    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=(
                    f"Please summarize the following document ('{doc_id}') in 2-3 concise sentences.\n\n"
                    f"Document content:\n{content}"
                ),
            ),
        )
    ]


@mcp.prompt(
    name="rewrite_as_markdown",
    description="Generate a prompt that asks the model to rewrite a document in clean Markdown.",
)
def prompt_rewrite_as_markdown(
    doc_id: str = Field(description="ID of the document to rewrite"),
) -> list[PromptMessage]:
    """
    Prompt: rewrite_as_markdown
    ---------------------------
    Demonstrates a prompt that asks the model to produce a *transformed*
    version of the content. The assistant message primes the model to reply
    with the reformatted document directly, without preamble.
    """
    if doc_id not in docs:
        raise ValueError(f"Document '{doc_id}' not found.")

    content = docs[doc_id]

    return [
        PromptMessage(
            role="user",
            content=TextContent(
                type="text",
                text=(
                    f"Rewrite the following document ('{doc_id}') using clean, well-structured Markdown.\n"
                    "Use headings, bullet points, and bold text where appropriate.\n\n"
                    f"Original content:\n{content}"
                ),
            ),
        ),
        # An assistant turn that steers the model to start its reply immediately.
        PromptMessage(
            role="assistant",
            content=TextContent(
                type="text",
                text=f"Here is the document rewritten in Markdown:\n\n",
            ),
        ),
    ]


# =============================================================================
# ENTRY POINT
# =============================================================================
# `mcp.run(transport="stdio")` starts the server and speaks the MCP protocol
# over stdin/stdout. This is the standard transport for locally-spawned servers.
#
# The client (MCPClient in mcp_client.py) launches this script as a subprocess
# and wires up the pipes automatically via `stdio_client()`.
#
# Other transports (HTTP/SSE, WebSocket) exist for remote servers but stdio is
# the simplest for local development.
# =============================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
