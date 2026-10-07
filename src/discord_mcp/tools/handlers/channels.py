from typing import Any, Dict, Iterable, List, Optional, Sequence

import discord
from mcp.types import TextContent

from discord_mcp.core.emoji import parse_emoji


def _json_text(message: str) -> List[TextContent]:
    return [TextContent(type="text", text=message)]


def _get_channel_type(channel: Any) -> str:
    channel_type = getattr(channel, "type", None)
    if isinstance(channel_type, str):
        return channel_type.lower()
    if hasattr(channel_type, "name"):
        return str(channel_type.name).lower()
    return channel.__class__.__name__.lower().removesuffix("channel")


def _matches_channel_type(channel: Any, expected: str) -> bool:
    channel_name = channel.__class__.__name__.lower()
    if channel_name == f"{expected}channel":
        return True
    return _get_channel_type(channel) == expected


def _resolve_category(guild: Any, category_id: Optional[str]):
    if not category_id:
        return None
    if hasattr(guild, "get_channel"):
        return guild.get_channel(int(category_id))
    return None


def _resolve_channel(guild: Any, channel_id: str):
    if hasattr(guild, "get_channel"):
        channel = guild.get_channel(int(channel_id))
        if channel is not None:
            return channel
    for channel in getattr(guild, "channels", []):
        if str(getattr(channel, "id", "")) == str(channel_id):
            return channel
    raise ValueError(f"Channel '{channel_id}' not found")


def _unsupported_fields(arguments: Dict[str, Any], allowed: Sequence[str]) -> List[str]:
    reserved = {"server_id", "channel_id", "reason"}
    return sorted(
        key for key in arguments if key not in allowed and key not in reserved
    )


async def _edit_channel(channel: Any, updates: Dict[str, Any], reason: Optional[str]):
    if not updates:
        return channel
    if reason is not None:
        updates["reason"] = reason
    await channel.edit(**updates)
    return channel


def _create_result(prefix: str, channel: Any, kind: str) -> List[TextContent]:
    return _json_text(f"{prefix} {kind} channel #{channel.name} (ID: {channel.id})")


async def handle_create_text_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    category = _resolve_category(guild, arguments.get("category_id"))

    channel = await guild.create_text_channel(
        name=arguments["name"],
        category=category,
        topic=arguments.get("topic"),
        reason="Channel created via MCP",
    )
    return _create_result("Created", channel, "text")


async def handle_create_voice_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    category = _resolve_category(guild, arguments.get("category_id"))

    channel = await guild.create_voice_channel(
        name=arguments["name"],
        category=category,
        bitrate=arguments.get("bitrate"),
        user_limit=arguments.get("user_limit"),
        rtc_region=arguments.get("rtc_region"),
        video_quality_mode=arguments.get("video_quality_mode"),
        reason=arguments.get("reason", "Channel created via MCP"),
    )
    return _create_result("Created", channel, "voice")


def _build_forum_tags(guild: Any, value: Any, where: str) -> List[Any]:
    """Build discord.ForumTag objects: ForumChannel.edit calls tag.to_dict()."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{where} must be an array of tag objects")

    tags = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"{where}[{index}] must be an object")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError(f"{where}[{index}].name is required")
        if len(name) > 20:
            raise ValueError(
                f"{where}[{index}].name is {len(name)} characters; Discord allows 20"
            )
        tags.append(
            discord.ForumTag(
                name=name,
                emoji=parse_emoji(guild, entry),
                moderated=bool(entry.get("moderated", False)),
            )
        )
    return tags


def _forum_default_reaction(guild: Any, value: Any) -> Optional[str]:
    """A custom emoji must keep its id here too, or Discord answers Invalid emoji id or name."""
    return parse_emoji(guild, value)


def _pop_forum_extras(guild: Any, source: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the read shape (camelCase) into discord.py's channel-edit keywords."""
    extras: Dict[str, Any] = {}
    tags = source.get("available_tags", source.get("availableTags"))
    if tags is not None:
        extras["available_tags"] = _build_forum_tags(guild, tags, "available_tags")
    reaction = source.get("default_reaction_emoji", source.get("defaultReactionEmoji"))
    if reaction is not None or "defaultReactionEmoji" in source:
        extras["default_reaction_emoji"] = _forum_default_reaction(guild, reaction)
    return extras


async def handle_create_forum_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    category = _resolve_category(guild, arguments.get("category_id"))

    extras = _pop_forum_extras(guild, arguments)
    channel = await guild.create_forum(
        name=arguments["name"],
        category=category,
        topic=arguments.get("topic"),
        nsfw=arguments.get("nsfw"),
        slowmode_delay=arguments.get("slowmode_delay"),
        default_auto_archive_duration=arguments.get("default_auto_archive_duration"),
        default_reaction_emoji=extras.get("default_reaction_emoji"),
        default_sort_order=arguments.get("default_sort_order"),
        available_tags=extras.get("available_tags"),
        reason=arguments.get("reason", "Channel created via MCP"),
    )
    return _create_result("Created", channel, "forum")


async def handle_get_channels(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    try:
        guild = await gateway.resolve_guild(arguments["server_id"])
        if guild:
            channel_list = [
                f"#{channel.name} (ID: {channel.id}) - {channel.type}"
                for channel in guild.channels
            ]
            return [
                TextContent(
                    type="text",
                    text=f"Channels in {guild.name}:\n" + "\n".join(channel_list),
                )
            ]
        return [TextContent(type="text", text="Guild not found")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_delete_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    channel = await gateway.fetch_channel(arguments["channel_id"])
    await channel.delete(reason=arguments.get("reason", "Channel deleted via MCP"))
    return [TextContent(type="text", text="Deleted channel successfully")]


async def handle_update_text_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    channel = _resolve_channel(guild, arguments["channel_id"])
    if not _matches_channel_type(channel, "text"):
        raise ValueError(f"Channel '{arguments['channel_id']}' is not a text channel")

    unsupported = _unsupported_fields(
        arguments,
        ["name", "topic", "nsfw", "slowmode_delay", "category_id", "position"],
    )
    if unsupported:
        raise ValueError(f"unsupported_fields: {', '.join(unsupported)}")

    updates: Dict[str, Any] = {}
    for key in ("name", "topic", "nsfw", "slowmode_delay", "position"):
        if key in arguments:
            updates[key] = arguments[key]
    if "category_id" in arguments:
        updates["category"] = _resolve_category(guild, arguments.get("category_id"))

    await _edit_channel(channel, updates, arguments.get("reason"))
    return _json_text(f"Updated text channel #{channel.name} (ID: {channel.id})")


async def handle_update_voice_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    channel = _resolve_channel(guild, arguments["channel_id"])
    if not _matches_channel_type(channel, "voice"):
        raise ValueError(f"Channel '{arguments['channel_id']}' is not a voice channel")

    unsupported = _unsupported_fields(
        arguments,
        [
            "name",
            "bitrate",
            "user_limit",
            "rtc_region",
            "video_quality_mode",
            "category_id",
            "position",
        ],
    )
    if unsupported:
        raise ValueError(f"unsupported_fields: {', '.join(unsupported)}")

    updates: Dict[str, Any] = {}
    for key in (
        "name",
        "bitrate",
        "user_limit",
        "rtc_region",
        "video_quality_mode",
        "position",
    ):
        if key in arguments:
            updates[key] = arguments[key]
    if "category_id" in arguments:
        updates["category"] = _resolve_category(guild, arguments.get("category_id"))

    await _edit_channel(channel, updates, arguments.get("reason"))
    return _json_text(f"Updated voice channel #{channel.name} (ID: {channel.id})")


async def handle_update_forum_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    channel = _resolve_channel(guild, arguments["channel_id"])
    if not _matches_channel_type(channel, "forum"):
        raise ValueError(f"Channel '{arguments['channel_id']}' is not a forum channel")

    unsupported = _unsupported_fields(
        arguments,
        [
            "name",
            "topic",
            "nsfw",
            "slowmode_delay",
            "default_auto_archive_duration",
            "default_reaction_emoji",
            "defaultReactionEmoji",
            "default_sort_order",
            "default_sort_order_value",
            "available_tags",
            "availableTags",
            "category_id",
            "position",
        ],
    )
    if unsupported:
        raise ValueError(f"unsupported_fields: {', '.join(unsupported)}")

    updates: Dict[str, Any] = {}
    for key in (
        "name",
        "topic",
        "nsfw",
        "slowmode_delay",
        "default_auto_archive_duration",
        "default_reaction_emoji",
        "default_sort_order",
        "available_tags",
        "position",
    ):
        if key in arguments:
            updates[key] = arguments[key]
    # emoji-bearing fields go through the shared codec so a read payload round trips
    updates.pop("available_tags", None)
    updates.pop("default_reaction_emoji", None)
    updates.update(_pop_forum_extras(guild, arguments))
    if "category_id" in arguments:
        updates["category"] = _resolve_category(guild, arguments.get("category_id"))

    await _edit_channel(channel, updates, arguments.get("reason"))
    return _json_text(f"Updated forum channel #{channel.name} (ID: {channel.id})")
