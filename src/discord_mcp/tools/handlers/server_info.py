from typing import Any, Dict, List

from mcp.types import TextContent


async def handle_get_server_info(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    info = {
        "name": guild.name,
        "id": str(guild.id),
        "owner_id": str(guild.owner_id),
        "member_count": guild.member_count,
        "created_at": guild.created_at.isoformat(),
        "description": guild.description,
        "premium_tier": guild.premium_tier,
        "explicit_content_filter": str(guild.explicit_content_filter),
    }
    return [
        TextContent(
            type="text",
            text="Server Information:\n"
            + "\n".join(f"{k}: {v}" for k, v in info.items()),
        )
    ]


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


async def handle_list_members(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    limit = min(int(arguments.get("limit", 100)), 1000)

    members = []
    async for member in guild.fetch_members(limit=limit):
        members.append(
            {
                "id": str(member.id),
                "name": member.name,
                "nick": member.nick,
                "joined_at": member.joined_at.isoformat() if member.joined_at else None,
                "roles": [str(role.id) for role in member.roles[1:]],
            }
        )

    return [
        TextContent(
            type="text",
            text=f"Server Members ({len(members)}):\n"
            + "\n".join(
                f"{m['name']} (ID: {m['id']}, Roles: {', '.join(m['roles'])})"
                for m in members
            ),
        )
    ]


async def handle_list_servers(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    servers = [
        {
            "id": str(guild.id),
            "name": guild.name,
            "member_count": guild.member_count,
            "created_at": guild.created_at.isoformat(),
        }
        for guild in gateway.client.guilds
    ]

    return [
        TextContent(
            type="text",
            text=f"Available Servers ({len(servers)}):\n"
            + "\n".join(
                f"{s['name']} (ID: {s['id']}, Members: {s['member_count']})"
                for s in servers
            ),
        )
    ]
