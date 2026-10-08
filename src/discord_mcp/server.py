import os
import sys
import asyncio
import logging
from typing import Any, List
from functools import wraps

import discord
from discord.ext import commands
from mcp.server import Server, ServerRequestContext
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    ServerCapabilities,
    TextContent,
    ToolsCapability,
)

from ._version import __version__
from .composition import (
    build_tool_dependencies,
    compose_tool_registry,
    dispatch_tool_call,
)


def _configure_windows_stdout_encoding():
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


_configure_windows_stdout_encoding()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord-mcp-server")

# Lazy token resolver: validated at runtime, not import time
def _require_discord_token() -> str:
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN environment variable is required")
    return token


# Initialize Discord bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Store Discord client reference
discord_client = None


@bot.event
async def on_ready():
    global discord_client
    discord_client = bot
    logger.info(f"Logged in as {bot.user.name}")


def _require_ready_client() -> None:
    """Raise unless the Discord gateway finished connecting."""
    if not discord_client:
        raise RuntimeError("Discord client not ready")


@wraps(dispatch_tool_call)
async def _run_tool(name: str, arguments: Any) -> List[TextContent]:
    """Dispatch one tool against a freshly built dependency set."""
    _require_ready_client()
    return await dispatch_tool_call(name, arguments, build_tool_dependencies(discord_client))


async def _on_call_tool(
    context: ServerRequestContext, params: CallToolRequestParams
) -> CallToolResult:
    """Run one Discord tool call.

    MCP SDK 2.x replaced the ``@app.call_tool()`` decorator with a constructor
    callback that receives typed request params instead of ``(name, arguments)``.

    A failing tool comes back as a result with ``is_error`` set rather than as a
    transport fault: SDK 2.x turns an escaping exception into a JSON-RPC error,
    which would hide "unknown tool" / "channel not found" / "missing permission"
    behind an opaque connection failure instead of text the model can read.
    """
    try:
        content = await _run_tool(params.name, params.arguments or {})
        return CallToolResult(content=content)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as tool output
        logger.warning("tool %s failed: %s", params.name, exc)
        return CallToolResult(
            content=[TextContent(type="text", text=f"{type(exc).__name__}: {exc}")],
            is_error=True,
        )


async def _on_list_tools(context: ServerRequestContext, params: Any) -> ListToolsResult:
    """Advertise the Discord tool registry."""
    return ListToolsResult(tools=compose_tool_registry())


# Initialize MCP server (the callbacks must exist before the Server is built)
app: Server = Server(
    name="discord-server",
    version=__version__,
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
)


async def main():
    # Validate token at runtime (not import time)
    token = _require_discord_token()

    # Start Discord bot in the background
    asyncio.create_task(bot.start(token))

    # Build explicit initialization options for the MCP server
    init_opts = InitializationOptions(
        server_name="discord-server",
        server_version=__version__,
        capabilities=ServerCapabilities(
            tools=ToolsCapability(listChanged=False),
        ),
    )

    # Run MCP server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, init_opts)


if __name__ == "__main__":
    asyncio.run(main())