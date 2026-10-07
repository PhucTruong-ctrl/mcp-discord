import json
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.emoji import serialize_emoji


def _emoji_row(emoji: Any) -> Dict[str, Any]:
    name, emoji_id, animated = serialize_emoji(emoji)
    row: Dict[str, Any] = {
        "id": emoji_id,
        "name": name,
        "animated": animated,
        "available": bool(getattr(emoji, "available", True)),
        "managed": bool(getattr(emoji, "managed", False)),
        "requireColons": bool(getattr(emoji, "require_colons", True)),
        "token": f"<{'a' if animated else ''}:{name}:{emoji_id}>",
        "url": str(getattr(emoji, "url", "") or "") or None,
    }
    roles = getattr(emoji, "roles", None)
    if roles is not None:
        row["roleIds"] = [str(getattr(role, "id", role)) for role in roles]
    return row


def _sticker_row(sticker: Any) -> Dict[str, Any]:
    return {
        "id": str(sticker.id),
        "name": getattr(sticker, "name", None),
        "description": getattr(sticker, "description", None),
        "type": str(getattr(sticker, "type", None)),
        "format": str(getattr(sticker, "format", None)),
        "available": bool(getattr(sticker, "available", True)),
        "tags": str(getattr(sticker, "tags", "") or ""),
    }


async def handle_list_guild_emojis(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    """List the guild's custom emoji (ids usable by the reaction tools) and stickers."""
    guild = await deps["gateway"].resolve_guild(arguments["server_id"])
    include_stickers = bool(arguments.get("include_stickers", False))
    name_filter = arguments.get("name_contains")

    emojis = [
        _emoji_row(emoji)
        for emoji in sorted(
            getattr(guild, "emojis", ()) or (),
            key=lambda item: str(getattr(item, "name", "")),
        )
    ]
    if name_filter:
        needle = str(name_filter).lower()
        emojis = [row for row in emojis if needle in (row["name"] or "").lower()]

    stickers = (
        [
            _sticker_row(sticker)
            for sticker in sorted(
                getattr(guild, "stickers", ()) or (),
                key=lambda item: str(getattr(item, "name", "")),
            )
        ]
        if include_stickers
        else []
    )

    payload = {
        "serverId": str(guild.id),
        "emojiCount": len(emojis),
        "emojis": emojis,
        "stickerCount": len(stickers),
        "stickers": stickers,
        "usage": (
            "Reaction tools accept either the unicode emoji or the custom emoji id; "
            "token is the <:name:id> / <a:name:id> form seen in message content."
        ),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]
