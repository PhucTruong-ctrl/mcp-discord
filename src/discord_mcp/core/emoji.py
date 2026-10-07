"""Shared emoji codec: read a discord.py emoji, write a Discord-parseable token.

Custom emoji are identified by *id*; a bare name is not enough for Discord (it answers
`Invalid emoji id or name`). Every tool that reads or writes an emoji field routes through
these two helpers so a read -> write round trip never loses the id or the animated flag.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import discord

from discord_mcp.core.resolve import try_int


def serialize_emoji(emoji: Any) -> Tuple[Optional[str], Optional[str], bool]:
    """Return ``(name, id, animated)`` for any emoji-ish value.

    Unicode emoji come back as ``("🔥", None, False)``; custom ones carry their id so the
    write path can rebuild ``<:name:id>`` / ``<a:name:id>``.
    """
    if not emoji:
        return (None, None, False)
    if isinstance(emoji, str):
        return (emoji, None, False)
    name = getattr(emoji, "name", None) or str(emoji)
    emoji_id = getattr(emoji, "id", None)
    return (
        str(name),
        str(emoji_id) if emoji_id else None,
        bool(getattr(emoji, "animated", False)),
    )


def emoji_payload(emoji: Any) -> Dict[str, Any]:
    """JSON shape shared by every emoji field: name plus the id/animated needed to write it."""
    name, emoji_id, animated = serialize_emoji(emoji)
    return {"emoji": name, "emojiId": emoji_id, "emojiAnimated": animated}


def emoji_token(name: Any, emoji_id: Any, animated: bool = False) -> Optional[str]:
    """Build the token discord.py can parse (unicode text, or ``<:name:id>``/``<a:name:id>``)."""
    if not name:
        return None
    text = str(name)
    snowflake = try_int(emoji_id)
    if snowflake is None:
        return text
    return f"<{'a' if animated else ''}:{text}:{snowflake}>"


def parse_emoji(guild: Any, value: Any) -> Optional[str]:
    """Turn any accepted emoji reference into a token Discord accepts.

    Accepts ``None``, a unicode emoji, a bare custom-emoji name (resolved against
    ``guild.emojis``), an explicit ``<:name:id>`` / ``<a:name:id>`` token, a dict
    ``{name, id, animated}``, or the read shape ``{emoji, emojiId, emojiAnimated}``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("<") and text.endswith(">"):
            return text  # already a token
        # a bare name only means a custom emoji if the guild actually has it
        resolved = _lookup_guild_emoji(guild, text)
        if resolved is not None:
            return resolved
        return text  # unicode emoji or unknown name: hand it to Discord as-is

    if isinstance(value, dict):
        name = value.get("emoji", value.get("name"))
        emoji_id = value.get("emojiId", value.get("id"))
        animated = bool(value.get("emojiAnimated", value.get("animated", False)))
        # the raw API shape nests the emoji object: {"emoji": {"id", "name", "animated"}}
        if isinstance(name, dict):
            emoji_id = name.get("id", emoji_id)
            animated = bool(name.get("animated", animated))
            name = name.get("name")
        if isinstance(name, str) and name.startswith("<") and name.endswith(">"):
            return name  # already a token
        if not name and emoji_id is None:
            return None
        if emoji_id is None:
            return parse_emoji(guild, name)
        return emoji_token(name, emoji_id, animated)

    # discord.py emoji object
    name, emoji_id, animated = serialize_emoji(value)
    return emoji_token(name, emoji_id, animated)


def parse_welcome_emoji(guild: Any, value: Any) -> Any:
    """Emoji value for ``discord.WelcomeChannel``.

    ``WelcomeChannel.to_dict()`` only fills ``emoji_id`` when the emoji is an object
    (``_EmojiTag``); a ``<:name:id>`` string would be sent as ``emoji_name`` alone and Discord
    answers "Invalid emoji id or name". Unicode emoji stay plain strings.
    """
    token = parse_emoji(guild, value)
    if not token:
        return None
    if isinstance(token, str) and token.startswith("<") and token.endswith(">"):
        return discord.PartialEmoji.from_str(token)
    return token


def _lookup_guild_emoji(guild: Any, name: str) -> Optional[str]:
    if guild is None:
        return None
    for emoji in getattr(guild, "emojis", ()) or ():
        if getattr(emoji, "name", None) == name:
            return emoji_token(
                name,
                getattr(emoji, "id", None),
                bool(getattr(emoji, "animated", False)),
            )
    return None
