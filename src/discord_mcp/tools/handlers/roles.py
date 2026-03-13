from typing import Any, Dict, List

from mcp.types import TextContent


async def handle_add_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.fetch_guild(arguments["server_id"])
    member = await guild.fetch_member(int(arguments["user_id"]))
    role = guild.get_role(int(arguments["role_id"]))
    await member.add_roles(role, reason="Role added via MCP")
    return [
        TextContent(type="text", text=f"Added role {role.name} to user {member.name}")
    ]


async def handle_remove_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.fetch_guild(arguments["server_id"])
    member = await guild.fetch_member(int(arguments["user_id"]))
    role = guild.get_role(int(arguments["role_id"]))
    await member.remove_roles(role, reason="Role removed via MCP")
    return [
        TextContent(
            type="text", text=f"Removed role {role.name} from user {member.name}"
        )
    ]
