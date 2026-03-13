from typing import Any, Dict, List

from mcp.types import TextContent


async def handle_get_server_info(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["discord_client"].fetch_guild(int(arguments["server_id"]))
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


async def handle_list_servers(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    servers = []
    for guild in deps["discord_client"].guilds:
        servers.append(
            {
                "id": str(guild.id),
                "name": guild.name,
                "member_count": guild.member_count,
                "created_at": guild.created_at.isoformat(),
            }
        )

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
