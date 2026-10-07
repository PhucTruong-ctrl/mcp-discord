import json
from typing import Any, Dict, List

import discord
from mcp.types import TextContent

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
