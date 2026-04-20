from typing import Any, Dict, List

from mcp.types import TextContent


def _channel_tool_pending(name: str) -> List[TextContent]:
    raise NotImplementedError(f"{name} is not implemented")


async def handle_create_text_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    category = None
    if "category_id" in arguments:
        category = guild.get_channel(int(arguments["category_id"]))

    channel = await guild.create_text_channel(
        name=arguments["name"],
        category=category,
        topic=arguments.get("topic"),
        reason="Channel created via MCP",
    )
    return [
        TextContent(
            type="text", text=f"Created text channel #{channel.name} (ID: {channel.id})"
        )
    ]


async def handle_create_voice_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _channel_tool_pending("create_voice_channel")


async def handle_create_forum_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _channel_tool_pending("create_forum_channel")


async def handle_update_text_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _channel_tool_pending("update_text_channel")


async def handle_update_voice_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _channel_tool_pending("update_voice_channel")


async def handle_update_forum_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _channel_tool_pending("update_forum_channel")


async def handle_get_channels(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    try:
        guild = await gateway.resolve_guild(arguments["server_id"])
        if guild:
            channel_list = [
                f"#{channel.name} (ID: {channel.id}) - {channel.type}"
                for channel in guild.channels
            ]
            return [
                TextContent(
                    type="text",
                    text=f"Channels in {guild.name}:\n" + "\n".join(channel_list),
                )
            ]
        return [TextContent(type="text", text="Guild not found")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_delete_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    channel = await gateway.fetch_channel(arguments["channel_id"])
    await channel.delete(reason=arguments.get("reason", "Channel deleted via MCP"))
    return [TextContent(type="text", text="Deleted channel successfully")]
