from typing import Any, Dict, List

from mcp.types import TextContent, Tool

from discord_mcp.core.resolve import normalize_name, try_int
from discord_mcp.core.runtime import get_default_guild_id, get_max_archived_threads_scan
from discord_mcp.core.serialize import _serialize_forum_tag, _serialize_message
from discord_mcp.services.discord_gateway import DiscordGateway
from discord_mcp.tools.handlers.router import dispatch_tool_call as route_tool_call
from discord_mcp.tools.schemas import (
    compose_tool_registry as compose_schema_tool_registry,
)


def compose_tool_registry() -> List[Tool]:
    return compose_schema_tool_registry()


def build_tool_dependencies(discord_client: Any) -> Dict[str, Any]:
    gateway = DiscordGateway(lambda: discord_client, get_default_guild_id())
    return {
        "discord_client": discord_client,
        "gateway": gateway,
        "try_int": try_int,
        "normalize_name": normalize_name,
        "serialize_message": _serialize_message,
        "serialize_forum_tag": _serialize_forum_tag,
        "max_archived_threads_scan": get_max_archived_threads_scan(),
    }


async def dispatch_tool_call(
    name: str,
    arguments: Any,
    dependencies: Dict[str, Any],
) -> List[TextContent]:
    return await route_tool_call(name, arguments, dependencies)
