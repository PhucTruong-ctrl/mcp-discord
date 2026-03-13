from typing import Any, Dict, List

from mcp.types import TextContent


async def handle_list_members(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["discord_client"].fetch_guild(int(arguments["server_id"]))
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


async def handle_add_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["discord_client"].fetch_guild(int(arguments["server_id"]))
    member = await guild.fetch_member(int(arguments["user_id"]))
    role = guild.get_role(int(arguments["role_id"]))
    await member.add_roles(role, reason="Role added via MCP")
    return [
        TextContent(type="text", text=f"Added role {role.name} to user {member.name}")
    ]


async def handle_remove_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["discord_client"].fetch_guild(int(arguments["server_id"]))
    member = await guild.fetch_member(int(arguments["user_id"]))
    role = guild.get_role(int(arguments["role_id"]))
    await member.remove_roles(role, reason="Role removed via MCP")
    return [
        TextContent(
            type="text", text=f"Removed role {role.name} from user {member.name}"
        )
    ]
