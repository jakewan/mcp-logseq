"""End-to-end coverage of the served MCP protocol surface (issue #92).

Every other test in this suite drives the tool handlers directly, so the wiring
between them and the SDK — the part mcp 2.0 changed — went untested until a user
started the server. Here a real ``ClientSession`` talks to the real ``Server``
over in-memory streams: ``initialize``, ``tools/list`` and ``tools/call`` all go
through the genuine SDK request pipeline.
"""

from contextlib import asynccontextmanager
from functools import partial

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import TextContent, Tool

from mcp_logseq.server import build_app, _WRITE_TOOL_NAMES
from mcp_logseq.tools import ToolHandler


class StubToolHandler(ToolHandler):
    """A tool with a typed schema that needs no Logseq API behind it."""

    def __init__(self):
        super().__init__("stub_tool")

    def get_tool_description(self):
        return Tool(
            name=self.name,
            description="Echo the given text, or fail on demand.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

    def run_tool(self, args: dict):
        if args["text"] == "boom":
            raise RuntimeError("stub exploded")
        return [TextContent(type="text", text=f"echo: {args['text']}")]


@asynccontextmanager
async def _connected(read_only: bool = False):
    """Yield ``(session, handlers)`` for an initialized in-memory connection."""
    app, handlers = build_app(read_only=read_only)
    async with create_client_server_memory_streams() as (
        (client_read, client_write),
        (server_read, server_write),
    ):
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                partial(
                    app.run,
                    server_read,
                    server_write,
                    app.create_initialization_options(),
                    raise_exceptions=True,
                )
            )
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                yield session, handlers
            tg.cancel_scope.cancel()


@pytest.mark.asyncio
async def test_initialize_and_list_tools():
    async with _connected() as (session, handlers):
        result = await session.list_tools()

    assert [t.name for t in result.tools] == list(handlers)
    assert all(t.input_schema["type"] == "object" for t in result.tools)


@pytest.mark.asyncio
async def test_read_only_app_serves_no_write_tools():
    async with _connected(read_only=True) as (session, _):
        result = await session.list_tools()

    assert _WRITE_TOOL_NAMES.isdisjoint({t.name for t in result.tools})


# MCP's readOnlyHint defaults to false, so a client that gates writes (Claude
# Code's plan mode, for one) treats an unannotated tool as a write and prompts
# on every call. Every tool the server can serve must appear in one of these
# sets, so a new tool is classified on purpose rather than inheriting either
# default. sync_vector_db is a read: it only returns instructions for running
# the external logseq-sync CLI.
_EXPECTED_READ_TOOLS = frozenset(
    {
        "list_pages",
        "get_page_content",
        "get_block",
        "search",
        "query",
        "find_pages_by_property",
        "get_pages_from_namespace",
        "get_pages_tree_from_namespace",
        "get_page_backlinks",
        "vector_search",
        "vector_db_status",
        "sync_vector_db",
    }
)
_EXPECTED_WRITE_TOOLS = _WRITE_TOOL_NAMES
_VECTOR_TOOLS = frozenset({"vector_search", "vector_db_status", "sync_vector_db"})


def _enable_vector_tools(monkeypatch, tmp_path):
    """Register the vector tools, which build_app serves only when configured."""
    from mcp_logseq.config import EmbedderConfig, VectorConfig

    config = VectorConfig(
        enabled=True,
        db_path=str(tmp_path / "vector"),
        embedder=EmbedderConfig(provider="ollama", model="placeholder"),
        graph_path=str(tmp_path / "graph"),
    )
    monkeypatch.setattr("mcp_logseq.config.load_vector_config", lambda: config)


@pytest.mark.asyncio
@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("vector", [False, True])
async def test_tools_declare_whether_they_are_read_only(
    monkeypatch, tmp_path, read_only, vector
):
    monkeypatch.delenv("LOGSEQ_CONFIG_FILE", raising=False)
    if vector:
        _enable_vector_tools(monkeypatch, tmp_path)
    async with _connected(read_only=read_only) as (session, _):
        result = await session.list_tools()

    served = {tool.name for tool in result.tools}
    # Without this, a vector case that silently served no vector tools would
    # pass while checking nothing behind the gate.
    assert (served >= _VECTOR_TOOLS) if vector else served.isdisjoint(_VECTOR_TOOLS)

    for tool in result.tools:
        assert tool.name in _EXPECTED_READ_TOOLS | _EXPECTED_WRITE_TOOLS, (
            f"{tool.name} is not classified as read-only or write"
        )
        got = bool(tool.annotations and tool.annotations.read_only_hint)
        assert got == (tool.name in _EXPECTED_READ_TOOLS), (
            f"{tool.name} read_only_hint = {got}"
        )


@pytest.mark.asyncio
async def test_call_tool_returns_handler_content():
    async with _connected() as (session, handlers):
        handlers["stub_tool"] = StubToolHandler()
        result = await session.call_tool("stub_tool", {"text": "hello"})

    assert result.is_error is False
    assert result.content[0].text == "echo: hello"


@pytest.mark.asyncio
async def test_unknown_tool_is_reported_as_an_error_result():
    async with _connected() as (session, _):
        result = await session.call_tool("no_such_tool", {})

    assert result.is_error is True
    assert result.content[0].text == "Unknown tool: no_such_tool"


@pytest.mark.asyncio
async def test_handler_failure_reaches_the_client_verbatim():
    async with _connected() as (session, handlers):
        handlers["stub_tool"] = StubToolHandler()
        result = await session.call_tool("stub_tool", {"text": "boom"})

    assert result.is_error is True
    assert result.content[0].text == "Error: stub exploded"


@pytest.mark.asyncio
async def test_arguments_are_validated_against_the_tool_schema():
    async with _connected() as (session, handlers):
        handlers["stub_tool"] = StubToolHandler()
        missing = await session.call_tool("stub_tool", {})
        mistyped = await session.call_tool("stub_tool", {"text": 42})

    assert missing.is_error is True
    assert missing.content[0].text.startswith("Input validation error:")
    assert mistyped.is_error is True
    assert "42 is not of type 'string'" in mistyped.content[0].text
