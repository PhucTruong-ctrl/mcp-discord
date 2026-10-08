from typing import Any, Dict, List, Optional

import discord
from discord.utils import resolve_invite
from mcp.types import TextContent

from discord_mcp.core.common import (
    as_id,
    json_text,
    member_row,
    require_gateway,
    user_row,
)
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.validation import (
    require_reason,
    validate_enum,
    validate_limit,
    validate_snowflake,
)


def _int(value: Any, field: str, default: Any = 0) -> int:
    if value is None:
        value = default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be an integer, got {value!r}") from None


def _snowflake(value: Any, field: str, server_id: str) -> int:
    try:
        return validate_snowflake(value)
    except ValueError:
        raise ValueError(
            f"{field} '{value}' is not a valid snowflake for server '{server_id}'"
        ) from None


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if hasattr(value, "isoformat") else None


async def _resolve_invite_channel(
    gateway: Any, guild: Any, channel_id: Any, server_id: str
):
    """Resolve an invite-capable channel (mirrors resolve_text_or_thread_channel)."""
    channel_id_int = _snowflake(channel_id, "channel_id", server_id)
    channel = guild.get_channel(channel_id_int)
    if channel is None:
        try:
            channel = await gateway.fetch_channel(str(channel_id_int))
        except discord.NotFound:
            channel = None
    if channel is None:
        raise ValueError(f"Channel '{channel_id}' not found in server '{server_id}'")
    owner_id = getattr(getattr(channel, "guild", None), "id", None)
    if owner_id is not None and str(owner_id) != str(guild.id):
        raise ValueError(
            f"Channel '{channel_id}' does not belong to server '{server_id}'"
        )
    if not (hasattr(channel, "invites") and hasattr(channel, "create_invite")):
        raise ValueError(
            f"Channel '{channel_id}' in server '{server_id}' does not support invites"
        )
    return channel


def _invite_row(invite: Any) -> Dict[str, Any]:
    channel = getattr(invite, "channel", None)
    inviter = getattr(invite, "inviter", None)
    target_type = getattr(invite, "target_type", None)
    return {
        "code": getattr(invite, "code", None),
        "url": getattr(invite, "url", None),
        "channelId": as_id(getattr(channel, "id", None)),
        "channelName": getattr(channel, "name", None),
        "inviter": user_row(inviter) if inviter is not None else None,
        "uses": getattr(invite, "uses", None),
        "maxUses": getattr(invite, "max_uses", None),
        "maxAge": getattr(invite, "max_age", None),
        "temporary": getattr(invite, "temporary", None),
        "createdAt": _iso(getattr(invite, "created_at", None)),
        "expiresAt": _iso(getattr(invite, "expires_at", None)),
        "targetType": getattr(target_type, "name", None),
    }


async def handle_create_invite(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_invite")
    guild = await gateway.resolve_guild(arguments["server_id"])
    server_id = str(guild.id)

    max_age = _int(arguments.get("max_age"), "max_age")
    max_uses = _int(arguments.get("max_uses"), "max_uses")
    temporary = bool(arguments.get("temporary", False))
    unique = bool(arguments.get("unique", True))
    guest = bool(arguments.get("guest", False))
    reason = str(arguments.get("reason") or "").strip() or None

    target_type = None
    if arguments.get("target_type"):
        name = validate_enum(
            str(arguments["target_type"]),
            ["stream", "embedded_application"],
            "target_type",
        )
        target_type = discord.InviteTarget[name]

    target_user = None
    if arguments.get("target_user"):
        target_user = discord.Object(
            id=_snowflake(arguments["target_user"], "target_user", server_id)
        )

    target_application_id = None
    if arguments.get("target_application_id"):
        target_application_id = _snowflake(
            arguments["target_application_id"], "target_application_id", server_id
        )

    channel = await _resolve_invite_channel(
        gateway, guild, arguments["channel_id"], server_id
    )

    action = "create_invite"
    targets = {
        "channel_id": str(channel.id),
        "server_id": server_id,
        "max_age": max_age,
        "max_uses": max_uses,
        "temporary": temporary,
        "unique": unique,
        "guest": guest,
        "target_type": target_type.name if target_type else None,
        "target_user": str(target_user.id) if target_user else None,
        "target_application_id": (
            str(target_application_id) if target_application_id else None
        ),
    }
    details = {
        "serverId": server_id,
        "channelId": str(channel.id),
        "channelName": getattr(channel, "name", None),
        "maxAge": max_age,
        "maxUses": max_uses,
        "temporary": temporary,
        "unique": unique,
        "guest": guest,
        "targetType": target_type.name if target_type else None,
        "reason": reason or "",
    }
    if bool(arguments.get("dry_run", True)):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    invite = await channel.create_invite(
        max_age=max_age,
        max_uses=max_uses,
        temporary=temporary,
        unique=unique,
        target_type=target_type,
        target_user=target_user,
        target_application_id=target_application_id,
        guest=guest,
        reason=reason,
    )
    return json_text(
        {
            "status": "executed",
            "action": action,
            "invite": {
                "code": invite.code,
                "url": invite.url,
                "channelId": str(channel.id),
                "expiresAt": _iso(getattr(invite, "expires_at", None)),
                "maxUses": getattr(invite, "max_uses", None),
                "maxAge": getattr(invite, "max_age", None),
                "temporary": getattr(invite, "temporary", None),
            },
        }
    )


async def handle_list_invites(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "list_invites")
    guild = await gateway.resolve_guild(arguments["server_id"])
    server_id = str(guild.id)

    if arguments.get("channel_id"):
        channel = await _resolve_invite_channel(
            gateway, guild, arguments["channel_id"], server_id
        )
        invites = await channel.invites()
    else:
        invites = await guild.invites()

    rows = [_invite_row(invite) for invite in invites]
    return json_text({"serverId": server_id, "count": len(rows), "invites": rows})


async def handle_delete_invite(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_invite")
    reason = require_reason(arguments.get("reason"), "delete_invite")
    guild = await gateway.resolve_guild(arguments["server_id"])
    server_id = str(guild.id)

    raw_code = str(arguments["invite_code"])
    try:
        code = resolve_invite(raw_code).code
    except ValueError:
        raise ValueError(
            f"invite_code '{raw_code}' is not a valid invite code "
            f"or URL for server '{server_id}'"
        ) from None

    try:
        invite = await gateway.client.fetch_invite(code)
    except discord.NotFound as exc:
        raise ValueError(f"Invite '{code}' not found in server '{server_id}'") from exc

    invite_guild = getattr(invite, "guild", None)
    if invite_guild is None or str(getattr(invite_guild, "id", "")) != server_id:
        raise ValueError(f"Invite '{code}' does not belong to server '{server_id}'")

    action = "delete_invite"
    targets = {"invite_code": code, "reason": reason, "server_id": server_id}
    if bool(arguments.get("dry_run", True)):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {
                    "serverId": server_id,
                    "inviteCode": code,
                    "url": getattr(invite, "url", None),
                    "reason": reason,
                },
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    try:
        await invite.delete(reason=reason)
    except discord.NotFound as exc:
        raise ValueError(
            f"Invite '{code}' no longer exists in server '{server_id}'"
        ) from exc
    return json_text({"status": "executed", "action": action, "inviteCode": code})


async def handle_list_bans(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "list_bans")
    guild = await gateway.resolve_guild(arguments["server_id"])
    limit = validate_limit(arguments.get("limit"), 1000, 1000)
    kwargs: Dict[str, Any] = {"limit": limit}
    for field in ("before", "after"):
        if arguments.get(field) is not None:
            kwargs[field] = discord.Object(id=_snowflake(arguments[field], field, str(guild.id)))

    entries = [entry async for entry in guild.bans(**kwargs)]
    rows = [
        {
            "userId": str(entry.user.id),
            "userName": getattr(entry.user, "name", None),
            "reason": entry.reason,
        }
        for entry in entries
    ]
    return json_text({"serverId": str(guild.id), "count": len(rows), "bans": rows})


async def handle_get_ban(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_ban")
    guild = await gateway.resolve_guild(arguments["server_id"])
    server_id = str(guild.id)
    user_id = _snowflake(arguments["user_id"], "user_id", server_id)

    try:
        ban = await guild.fetch_ban(discord.Object(id=user_id))
    except discord.NotFound as exc:
        raise ValueError(
            f"User '{user_id}' is not banned in server '{server_id}'"
        ) from exc

    return json_text(
        {
            "serverId": server_id,
            "userId": str(user_id),
            "userName": getattr(ban.user, "name", None),
            "reason": ban.reason,
        }
    )


async def handle_search_members(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "search_members")
    guild = await gateway.resolve_guild(arguments["server_id"])
    server_id = str(guild.id)

    query = arguments.get("query")
    query = str(query) if query not in (None, "") else None
    limit = validate_limit(arguments.get("limit"), 5, 100)

    user_ids = None
    if arguments.get("user_ids") is not None:
        raw_ids = arguments["user_ids"]
        if not isinstance(raw_ids, (list, tuple)):
            raise ValueError(f"user_ids must be an array for server '{server_id}'")
        user_ids = [_snowflake(value, "user_ids", server_id) for value in raw_ids]

    members = await guild.query_members(query=query, limit=limit, user_ids=user_ids)
    return json_text(
        {
            "serverId": server_id,
            "query": query,
            "count": len(members),
            "members": [member_row(member) for member in members],
        }
    )


async def handle_get_role_member_counts(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_role_member_counts")
    guild = await gateway.resolve_guild(arguments["server_id"])

    counts = await guild.role_member_counts()
    rows = [
        {
            "roleId": str(role.id),
            "roleName": getattr(role, "name", None),
            "position": getattr(role, "position", None),
            "count": count,
        }
        for role, count in counts.items()
    ]
    rows.sort(key=lambda row: (row["position"] is None, -(row["position"] or 0)))
    return json_text({"serverId": str(guild.id), "roles": rows})


async def handle_estimate_pruned_members(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "estimate_pruned_members")
    guild = await gateway.resolve_guild(arguments["server_id"])
    server_id = str(guild.id)

    days = _int(arguments.get("days"), "days", None)
    if not 1 <= days <= 30:
        raise ValueError(
            f"days must be between 1 and 30 for server '{server_id}', got {days}"
        )

    raw_role_ids = arguments.get("role_ids")
    if raw_role_ids is None:
        raw_role_ids = []
    if not isinstance(raw_role_ids, (list, tuple)):
        raise ValueError(f"role_ids must be an array for server '{server_id}'")
    role_ids = sorted(
        {_snowflake(value, "role_ids", server_id) for value in raw_role_ids}
    )

    count = await guild.estimate_pruned_members(
        days=days, roles=[discord.Object(id=role_id) for role_id in role_ids]
    )
    return json_text(
        {
            "serverId": server_id,
            "days": days,
            "roleIds": [str(role_id) for role_id in role_ids],
            "prunable": count is not None,
            "count": count,
        }
    )
