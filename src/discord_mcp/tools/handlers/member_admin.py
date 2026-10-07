import json
from typing import Any, Dict, List

import discord
from mcp.types import TextContent

from discord_mcp.core.resolve import try_int
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token

# Discord's per-guild nickname field is limited to 32 characters
NICKNAME_MAX_LENGTH = 32


def _nickname_value(raw: Any) -> Any:
    """Map the caller value onto discord.py's nickname semantics.

    discord.py's ``Member.edit(nick=...)`` takes ``None`` to remove the nickname;
    an empty string is treated the same way here so callers can clear it either way.
    """
    if raw is None:
        return None
    nickname = str(raw)
    if nickname == "":
        return None
    if len(nickname) > NICKNAME_MAX_LENGTH:
        raise ValueError(
            f"nickname must be at most {NICKNAME_MAX_LENGTH} characters "
            f"(got {len(nickname)})"
        )
    return nickname


async def _resolve_roles(guild: Any, role_ids: Any) -> List[Any]:
    """Resolve every id, rejecting roles Discord cannot assign manually."""
    if role_ids is None:
        return []
    if not isinstance(role_ids, (list, tuple)):
        raise ValueError("role_ids must be an array of role ids")

    resolved = []
    ignored = []
    for value in role_ids:
        snowflake = try_int(value)
        if snowflake is None:
            raise ValueError(f"role_ids entries must be snowflake ids, got {value!r}")
        if snowflake == getattr(guild, "id", None):
            # @everyone is implicit on every member and cannot be granted or removed
            ignored.append(str(snowflake))
            continue
        role = guild.get_role(snowflake)
        if role is None:
            fetch_roles = getattr(guild, "fetch_roles", None)
            if fetch_roles is not None:
                roles = await fetch_roles()
                role = next((item for item in roles if item.id == snowflake), None)
        if role is None:
            raise ValueError(f"Role '{value}' not found in server '{guild.id}'")
        if getattr(role, "managed", False):
            raise ValueError(
                f"Role '{role.name}' ({role.id}) is managed by an integration/bot and cannot "
                "be assigned manually"
            )
        resolved.append(role)
    return resolved, ignored


async def handle_set_member_roles(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    """Replace a member's role set in one call (dry_run + confirm_token gated)."""
    gateway = deps["gateway"]
    server_id = str(arguments["server_id"])
    member_id = str(arguments["member_id"])
    reason = str(arguments.get("reason", "")).strip() or None
    action = "set_member_roles"

    guild = await gateway.resolve_guild(server_id)
    try:
        member = await guild.fetch_member(int(member_id))
    except Exception as exc:  # noqa: BLE001 - surfaced with the member id
        raise ValueError(f"Member '{member_id}' not found in server {server_id}: {exc}")

    resolved, ignored = await _resolve_roles(guild, arguments.get("role_ids"))
    target_ids = sorted(str(role.id) for role in resolved)

    current_ids = await gateway.fetch_member_role_ids(server_id, member_id)
    current = {str(role_id) for role_id in current_ids}
    if getattr(guild, "id", None) is not None:
        current.discard(str(guild.id))  # @everyone is implicit

    # Roles owned by an integration (e.g. a bot's own role) cannot be removed by hand:
    # Discord answers 403 50013 if the request leaves them out, so keep them.
    preserved = []
    for role_id in sorted(current - set(target_ids)):
        role = guild.get_role(int(role_id))
        if role is not None and getattr(role, "managed", False):
            preserved.append(role)
    if preserved:
        resolved = resolved + preserved
        target_ids = sorted(str(role.id) for role in resolved)

    targets = {"server_id": server_id, "member_id": member_id, "role_ids": target_ids}
    added = sorted(set(target_ids) - current)
    removed = sorted(current - set(target_ids))

    if bool(arguments.get("dry_run", True)):
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    build_dry_run_result(
                        action,
                        targets,
                        {
                            "member": str(member),
                            "currentRoleIds": sorted(current),
                            "roleIds": target_ids,
                            "added": added,
                            "removed": removed,
                            "ignoredRoleIds": ignored,
                            "preservedManagedRoleIds": [
                                str(role.id) for role in preserved
                            ],
                            "reason": reason or "",
                        },
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        ]
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    try:
        await member.edit(roles=resolved, reason=reason)
    except discord.Forbidden as exc:
        top = getattr(getattr(guild, "me", None), "top_role", None)
        raise ValueError(
            f"Cannot replace roles for '{member_id}': {exc}. The bot needs MANAGE_ROLES and "
            f"its highest role must sit above every role being assigned "
            f"(bot top role: {getattr(top, 'name', None)!r} position {getattr(top, 'position', None)})."
        )

    final_ids = await gateway.fetch_member_role_ids(server_id, member_id)
    final = sorted(
        str(role_id)
        for role_id in final_ids
        if str(role_id) != str(getattr(guild, "id", ""))
    )
    payload = {
        "status": "executed",
        "action": action,
        "serverId": server_id,
        "memberId": member_id,
        "member": str(member),
        "requestedRoleIds": target_ids,
        "finalRoleIds": final,
        "added": added,
        "removed": removed,
        "ignoredRoleIds": ignored,
        "preservedManagedRoleIds": [str(role.id) for role in preserved],
        "matchesRequest": final == target_ids,
        "reason": reason,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_set_member_nickname(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = str(arguments["server_id"])
    member_id = str(arguments["member_id"])
    if "nickname" not in arguments:
        raise ValueError("nickname is required (pass an empty string to remove it)")
    nickname = _nickname_value(arguments.get("nickname"))
    reason = str(arguments.get("reason", "")).strip() or None

    guild = await gateway.resolve_guild(server_id)
    try:
        member = await guild.fetch_member(int(member_id))
    except Exception as exc:  # noqa: BLE001 - surfaced with the member id
        raise ValueError(f"Member '{member_id}' not found in server {server_id}: {exc}")

    previous = getattr(member, "nick", None)
    try:
        await member.edit(nick=nickname, reason=reason)
    except discord.Forbidden as exc:
        me = getattr(guild, "me", None)
        is_self = getattr(me, "id", None) == getattr(member, "id", None)
        required = "CHANGE_NICKNAME" if is_self else "MANAGE_NICKNAMES"
        raise ValueError(
            f"Cannot change the nickname of '{member_id}' in server {server_id}: missing "
            f"{required} permission (Discord said: {exc})"
        )

    payload = {
        "status": "executed",
        "action": "set_member_nickname",
        "serverId": server_id,
        "memberId": member_id,
        "member": str(member),
        "previousNickname": previous,
        "nickname": nickname,
        "nicknameRemoved": nickname is None,
        "displayName": getattr(member, "display_name", None),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]
