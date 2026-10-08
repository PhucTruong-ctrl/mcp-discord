import json
from typing import Any, Dict, List

import discord
from mcp.types import TextContent

from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token

async def _resolve_member(guild: Any, user_id: Any):
    try:
        return await guild.fetch_member(int(user_id))
    except Exception as exc:  # noqa: BLE001 - surfaced with the user id
        raise ValueError(f"Member '{user_id}' not found in server '{guild.id}': {exc}")


async def _resolve_role(guild: Any, role_id: Any):
    """Resolve a role by id, fetching the guild's roles if the cache misses."""
    parsed = None
    try:
        parsed = int(role_id)
    except (TypeError, ValueError):
        raise ValueError(f"role_id must be a snowflake id, got {role_id!r}")

    role = guild.get_role(parsed)
    if role is None:
        fetch_roles = getattr(guild, "fetch_roles", None)
        if fetch_roles is not None:
            roles = await fetch_roles()
            role = next((item for item in roles if item.id == parsed), None)
    if role is None:
        available = ", ".join(
            f"{getattr(item, 'name', '?')} ({item.id})"
            for item in list(getattr(guild, "roles", []) or [])[:15]
        )
        raise ValueError(
            f"Role '{role_id}' not found in server '{guild.id}'. Known roles: {available}"
        )
    return role


def _forbidden_hint(exc: Exception, role: Any, guild: Any) -> str:
    bot_top = getattr(getattr(guild, "me", None), "top_role", None)
    return (
        f"Cannot change role '{getattr(role, 'name', role.id)}' ({role.id}): {exc}. "
        "The bot needs MANAGE_ROLES, and its highest role must sit above the target role "
        f"(bot top role: {getattr(bot_top, 'name', None)!r}, position "
        f"{getattr(bot_top, 'position', None)} vs role position "
        f"{getattr(role, 'position', None)})."
    )


async def _set_role_membership(
    arguments: Dict[str, Any], deps: Dict[str, Any], *, add: bool
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    member = await _resolve_member(guild, arguments["user_id"])
    role = await _resolve_role(guild, arguments["role_id"])
    reason = str(arguments.get("reason") or "").strip()
    if not reason:
        raise ValueError("reason is required")

    server_id = str(arguments["server_id"])
    held_before = role.id in await gateway.fetch_member_role_ids(
        server_id, str(member.id)
    )

    targets = {
        "server_id": server_id,
        "user_id": str(member.id),
        "role_id": str(role.id),
        "reason": reason,
    }
    if bool(arguments.get("dry_run", True)):
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    build_dry_run_result(
                        "add_role" if add else "remove_role",
                        targets,
                        {
                            "serverId": str(guild.id),
                            "userId": str(member.id),
                            "roleId": str(role.id),
                            "roleName": role.name,
                            "reason": reason,
                            "hadRoleBefore": held_before,
                            "operation": "add" if add else "remove",
                        },
                    ),
                    ensure_ascii=False,
                ),
            )
        ]
    verify_confirm_token(
        "add_role" if add else "remove_role", targets, arguments.get("confirm_token")
    )

    try:
        if add:
            await member.add_roles(role, reason=reason)
        else:
            await member.remove_roles(role, reason=reason)
    except discord.Forbidden as exc:
        raise ValueError(_forbidden_hint(exc, role, guild))

    role_ids_after = await gateway.fetch_member_role_ids(server_id, str(member.id))
    held_after = role.id in role_ids_after
    payload = {
        "status": "executed",
        "action": "add_role" if add else "remove_role",
        "serverId": str(guild.id),
        "userId": str(member.id),
        "roleId": str(role.id),
        "roleName": role.name,
        "reason": reason,
        "hadRoleBefore": held_before,
        "hasRoleNow": held_after,
        "changed": held_before != held_after,
        "roleCount": len(role_ids_after),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_add_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return await _set_role_membership(arguments, deps, add=True)


async def handle_remove_role(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return await _set_role_membership(arguments, deps, add=False)
