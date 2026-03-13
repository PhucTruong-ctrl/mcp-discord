from typing import Any, Dict, List

from mcp.types import TextContent


async def handle_create_text_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.fetch_guild(arguments["server_id"])
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


async def handle_delete_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    channel = await gateway.fetch_channel(arguments["channel_id"])
    await channel.delete(reason=arguments.get("reason", "Channel deleted via MCP"))
    return [TextContent(type="text", text="Deleted channel successfully")]
