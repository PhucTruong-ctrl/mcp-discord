import json
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.permissions import (
    effective_permissions,
    overwrite_rows,
    permission_names,
    role_payload,
)


def _permission_block(bits: int) -> Dict[str, Any]:
    return {"permissions": bits, "permissionNames": permission_names(bits)}


async def handle_get_role_permissions(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    role_id = arguments.get("role_id")

    if role_id in (None, ""):
        roles = sorted(guild.roles, key=lambda role: role.position, reverse=True)
        payload = {
            "serverId": str(guild.id),
            "everyoneRoleId": str(
                getattr(getattr(guild, "default_role", None), "id", guild.id)
            ),
            "roleCount": len(roles),
            "roles": [role_payload(role) for role in roles],
        }
    else:
        role = guild.get_role(int(role_id))
        if role is None:
            raise ValueError(f"Role '{role_id}' not found in server '{guild.id}'")
        payload = {
            "serverId": str(guild.id),
            "roleCount": 1,
            "roles": [role_payload(role)],
        }

    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_compute_member_permissions(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    member_id = str(arguments["member_id"])
    member = await guild.fetch_member(int(member_id))

    channel = None
    channel_id = arguments.get("channel_id")
    if channel_id not in (None, ""):
        channel = guild.get_channel_or_thread(int(channel_id))
        if channel is None:
            channel = await gateway.fetch_channel(str(channel_id))

    resolved = effective_permissions(guild, member, channel)
    payload = {
        "serverId": str(guild.id),
        "memberId": member_id,
        "memberName": getattr(member, "display_name", None) or str(member),
        "channelId": str(channel.id) if channel is not None else None,
        "isOwner": resolved["isOwner"],
        "isAdministrator": resolved["isAdministrator"],
        "base": _permission_block(resolved["base"]),
        "effective": _permission_block(resolved["effective"]),
        # layer that decided each permission: base_role | administrator |
        # everyone_overwrite | role_overwrite:<roleId> | member_overwrite (:allow/:deny)
        "sources": resolved["sources"],
        "memberRoles": [
            role_payload(role) for role in getattr(member, "roles", []) or []
        ],
        "categoryOverwrites": _category_overwrites(channel),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


def _category_overwrites(channel: Any) -> List[Dict[str, Any]]:
    """Parent-category overwrites for a channel.

    Reported, not applied: Discord only inherits them into a channel once the
    channel is synced with its category, so they must not be folded into the
    effective bitfield above.
    """
    if channel is None:
        return []
    category = getattr(channel, "category", None)
    return overwrite_rows(category) if category is not None else []
