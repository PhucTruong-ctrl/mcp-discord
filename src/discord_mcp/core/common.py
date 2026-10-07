"""Shared handler helpers for the tool layer.

Small, boring primitives every tool handler needs: a gateway guard, a JSON
``TextContent`` wrapper, and common payload rows. Keep this module free of tool
knowledge so any domain can import it without creating a dependency edge.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mcp.types import TextContent

__all__ = [
    "json_text",
    "require_gateway",
    "as_id",
    "member_row",
    "channel_row",
    "user_row",
]


def json_text(payload: Dict[str, Any], *, indent: Optional[int] = None) -> List[TextContent]:
    """Wrap a payload as the single ``TextContent`` every tool returns."""
    return [
        TextContent(
            type="text",
            text=json.dumps(payload, ensure_ascii=False, indent=indent),
        )
    ]


def require_gateway(deps: Dict[str, Any], tool_name: str):
    """Return the live gateway or raise.

    Tools must never fall back to a synthetic payload when the gateway is
    missing: a bot that has not finished connecting cannot report guild state,
    and a fabricated answer is worse than an error.
    """
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError(f"gateway is required for {tool_name}")
    return gateway


def as_id(value: Any) -> Optional[str]:
    """Snowflakes cross the MCP boundary as strings; ``None`` stays ``None``."""
    if value is None or value == "":
        return None
    return str(value)


def _isoformat(value: Any) -> Optional[str]:
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else None


def user_row(user: Any) -> Dict[str, Any]:
    """Minimal user row: id, display name, bot flag, avatar."""
    avatar = getattr(user, "display_avatar", None)
    return {
        "id": as_id(getattr(user, "id", None)),
        "name": getattr(user, "name", None) or getattr(user, "username", None),
        "displayName": getattr(user, "display_name", None)
        or getattr(user, "global_name", None),
        "bot": bool(getattr(user, "bot", False)),
        "avatarUrl": str(avatar.url) if avatar is not None else None,
    }


def member_row(member: Any) -> Dict[str, Any]:
    """Member row shared by every member-emitting tool."""
    return {
        "id": as_id(getattr(member, "id", None)),
        "name": getattr(member, "display_name", None) or getattr(member, "name", None),
        "username": getattr(member, "name", None),
        "nick": getattr(member, "nick", None),
        "bot": bool(getattr(member, "bot", False)),
        "joinedAt": _isoformat(getattr(member, "joined_at", None)),
        "createdAt": _isoformat(getattr(member, "created_at", None)),
        "roleIds": [str(role.id) for role in getattr(member, "roles", []) or []],
        "timedOutUntil": _isoformat(getattr(member, "timed_out_until", None)),
    }


def channel_row(channel: Any) -> Dict[str, Any]:
    """Channel row shared by every channel-emitting tool."""
    return {
        "id": as_id(getattr(channel, "id", None)),
        "name": getattr(channel, "name", None),
        "type": str(getattr(channel, "type", "unknown")),
        "categoryId": as_id(getattr(channel, "category_id", None)),
        "position": getattr(channel, "position", None),
        "nsfw": getattr(channel, "nsfw", None),
        "archived": getattr(channel, "archived", None),
    }