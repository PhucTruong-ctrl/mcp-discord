"""Emoji, sticker and soundboard handlers (guild expressions and the soundboard)."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import discord
from mcp.types import TextContent

from discord_mcp.core.common import as_id, json_text, require_gateway
from discord_mcp.core.emoji import emoji_payload, parse_emoji
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.validation import require_reason, validate_snowflake
from discord_mcp.tools.handlers.emoji import _emoji_row

_URL_SCHEMES = ("http", "https")


def _is_dry_run(arguments: Dict[str, Any]) -> bool:
    return bool(arguments.get("dry_run", True))


def _client(deps: Dict[str, Any], tool: str) -> Any:
    client = deps.get("discord_client")
    if not client:
        raise ValueError(f"discord_client is required for {tool}")
    return client


def _download_url(url: Any, field: str) -> bytes:
    """Read an http(s) URL in-process; reject every other scheme outright."""
    text = str(url).strip()
    scheme = urlparse(text).scheme.lower()
    if scheme not in _URL_SCHEMES:
        raise ValueError(f"{field}: only http/https URLs are supported, got '{text}'")
    try:
        with urlopen(text, timeout=30) as response:  # noqa: S310 - scheme checked above
            return response.read()
    except (URLError, OSError) as exc:
        raise ValueError(f"{field}: failed to download '{text}': {exc}") from exc


async def _download(field: str, url: Any) -> bytes:
    return await asyncio.to_thread(_download_url, str(url), field)


def _name(
    arguments: Dict[str, Any], tool: str, *, required: bool = True
) -> Optional[str]:
    raw = arguments.get("name")
    if raw is None:
        if required:
            raise ValueError(f"name is required for {tool}")
        return None
    name = str(raw).strip()
    if not name:
        raise ValueError(f"name must not be empty for {tool}")
    return name


def _volume(arguments: Dict[str, Any]) -> Optional[float]:
    raw = arguments.get("volume")
    if raw is None:
        return None
    try:
        volume = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"volume must be a number between 0.0 and 1.0 (got {raw!r})"
        ) from None
    if not 0.0 <= volume <= 1.0:
        raise ValueError(f"volume must be between 0.0 and 1.0 (got {volume})")
    return volume


def _sticker_file_path(raw: Any) -> str:
    path = os.path.expanduser(str(raw or "").strip())
    if not path:
        raise ValueError("file_path is required for create_sticker")
    if not os.path.isfile(path):
        raise ValueError(f"file_path: '{path}' is not an existing file")
    return path


def _resolve_emoji(guild: Any, emoji_id: Any) -> Any:
    emoji_id_int = validate_snowflake(emoji_id)
    emoji = None
    getter = getattr(guild, "get_emoji", None)
    if getter is not None:
        emoji = getter(emoji_id_int)
    if emoji is None:
        for candidate in getattr(guild, "emojis", ()) or ():
            if getattr(candidate, "id", None) == emoji_id_int:
                emoji = candidate
                break
    if emoji is None:
        raise ValueError(f"Emoji '{emoji_id}' not found in server '{guild.name}'")
    return emoji


def _resolve_roles(guild: Any, role_ids: Any) -> Optional[List[Any]]:
    if role_ids is None:
        return None
    if isinstance(role_ids, (str, bytes)) or not isinstance(role_ids, (list, tuple)):
        raise ValueError("role_ids must be a list of role IDs")
    roles: List[Any] = []
    for raw in role_ids:
        role_id = validate_snowflake(raw)
        role = None
        getter = getattr(guild, "get_role", None)
        if getter is not None:
            role = getter(role_id)
        if role is None:
            for candidate in getattr(guild, "roles", ()) or ():
                if getattr(candidate, "id", None) == role_id:
                    role = candidate
                    break
        if role is None:
            raise ValueError(f"Role '{raw}' not found in server '{guild.name}'")
        roles.append(role)
    return roles


async def _resolve_sticker(guild: Any, sticker_id: Any) -> Any:
    """Cache first, then a direct API fetch (2.7.1 Guild has no ``get_sticker``)."""
    sticker_id_int = validate_snowflake(sticker_id)
    sticker = next(
        (
            item
            for item in getattr(guild, "stickers", ()) or ()
            if getattr(item, "id", None) == sticker_id_int
        ),
        None,
    )
    if sticker is None:
        try:
            sticker = await guild.fetch_sticker(sticker_id_int)
        except discord.NotFound:
            sticker = None
    if sticker is None:
        raise ValueError(f"Sticker '{sticker_id}' not found in server '{guild.name}'")
    return sticker


async def _resolve_sound(guild: Any, sound_id: Any) -> Any:
    sound_id_int = validate_snowflake(sound_id)
    getter = getattr(guild, "get_soundboard_sound", None)
    sound = getter(sound_id_int) if getter is not None else None
    if sound is None:
        fetched = await guild.fetch_soundboard_sounds()
        sound = next(
            (item for item in fetched if getattr(item, "id", None) == sound_id_int),
            None,
        )
    if sound is None:
        raise ValueError(f"Sound '{sound_id}' not found in server '{guild.name}'")
    return sound


async def _resolve_channel(guild: Any, channel_id: Any) -> Any:
    channel_id_int = validate_snowflake(channel_id)
    channel = None
    getter = getattr(guild, "get_channel", None)
    if getter is not None:
        channel = getter(channel_id_int)
    if channel is None:
        for candidate in getattr(guild, "channels", ()) or ():
            if getattr(candidate, "id", None) == channel_id_int:
                channel = candidate
                break
    if channel is None:
        fetch = getattr(guild, "fetch_channel", None)
        if fetch is not None:
            try:
                channel = await fetch(channel_id_int)
            except discord.NotFound:
                channel = None
    if channel is None:
        raise ValueError(f"Channel '{channel_id}' not found in server '{guild.name}'")
    channel_guild = getattr(channel, "guild", None)
    if channel_guild is not None and getattr(channel_guild, "id", None) != guild.id:
        raise ValueError(f"Channel '{channel_id}' is not in server '{guild.name}'")
    return channel


async def _fetch_application_emoji(client: Any, emoji_id: Any) -> Any:
    emoji_id_int = validate_snowflake(emoji_id)
    fetch = getattr(client, "fetch_application_emoji", None)
    if fetch is None:
        raise ValueError(
            f"Application emoji '{emoji_id}' cannot be resolved: the current runtime "
            "does not expose fetch_application_emoji"
        )
    try:
        return await fetch(emoji_id_int)
    except discord.NotFound:
        raise ValueError(f"Application emoji '{emoji_id}' not found") from None


def _sticker_row(sticker: Any) -> Dict[str, Any]:
    user = getattr(sticker, "user", None)
    return {
        "id": as_id(getattr(sticker, "id", None)),
        "name": getattr(sticker, "name", None),
        "description": getattr(sticker, "description", None),
        "emoji": getattr(sticker, "emoji", None),
        "format": str(getattr(sticker, "format", None)),
        "available": bool(getattr(sticker, "available", True)),
        "tags": str(getattr(sticker, "tags", "") or ""),
        "guildId": as_id(getattr(sticker, "guild_id", None)),
        "userId": as_id(getattr(user, "id", None)),
    }


def _sound_row(sound: Any) -> Dict[str, Any]:
    user = getattr(sound, "user", None)
    return {
        "id": as_id(getattr(sound, "id", None)),
        "name": getattr(sound, "name", None),
        "volume": getattr(sound, "volume", None),
        **emoji_payload(getattr(sound, "emoji", None)),
        "available": bool(getattr(sound, "available", True)),
        "userId": as_id(getattr(user, "id", None)),
        "url": str(getattr(sound, "url", "") or "") or None,
    }


# --- guild emoji ------------------------------------------------------------


async def handle_create_emoji(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_emoji")
    guild = await gateway.resolve_guild(arguments["server_id"])
    name = _name(arguments, "create_emoji")
    roles = _resolve_roles(guild, arguments.get("role_ids"))
    reason = arguments.get("reason")
    image = await _download("image_url", arguments["image_url"])

    action = "create_emoji"
    targets = {
        "server_id": str(guild.id),
        "name": name,
        "role_ids": sorted(str(role.id) for role in roles) if roles else [],
    }
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action, targets, {"serverId": str(guild.id), "name": name}
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs: Dict[str, Any] = {"name": name, "image": image}
    if roles:
        kwargs["roles"] = roles
    if reason is not None:
        kwargs["reason"] = reason
    emoji = await guild.create_custom_emoji(**kwargs)
    return json_text(
        {"status": "executed", "action": action, "emoji": _emoji_row(emoji)}
    )


async def handle_edit_emoji(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_emoji")
    guild = await gateway.resolve_guild(arguments["server_id"])
    emoji = _resolve_emoji(guild, arguments["emoji_id"])
    name = _name(arguments, "edit_emoji", required=False)
    roles = _resolve_roles(guild, arguments.get("role_ids"))
    reason = arguments.get("reason")

    action = "edit_emoji"
    targets = {"server_id": str(guild.id), "emoji_id": str(emoji.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"emojiId": str(emoji.id)})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs: Dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if roles is not None:
        kwargs["roles"] = roles
    if reason is not None:
        kwargs["reason"] = reason
    updated = await emoji.edit(**kwargs)
    return json_text(
        {"status": "executed", "action": action, "emoji": _emoji_row(updated)}
    )


async def handle_delete_emoji(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_emoji")
    guild = await gateway.resolve_guild(arguments["server_id"])
    emoji = _resolve_emoji(guild, arguments["emoji_id"])
    reason = require_reason(arguments.get("reason"), "delete_emoji")

    action = "delete_emoji"
    targets = {"server_id": str(guild.id), "emoji_id": str(emoji.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"emojiId": str(emoji.id)})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await guild.delete_emoji(emoji, reason=reason)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "serverId": str(guild.id),
            "emojiId": str(emoji.id),
        }
    )


# --- application emoji ------------------------------------------------------


async def handle_create_application_emoji(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_application_emoji")
    client = _client(deps, "create_application_emoji")
    name = _name(arguments, "create_application_emoji")
    image = await _download("image_url", arguments["image_url"])

    action = "create_application_emoji"
    targets = {"name": name}
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, {"name": name}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    emoji = await client.create_application_emoji(name=name, image=image)
    return json_text(
        {"status": "executed", "action": action, "emoji": _emoji_row(emoji)}
    )


async def handle_edit_application_emoji(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_application_emoji")
    client = _client(deps, "edit_application_emoji")
    emoji = await _fetch_application_emoji(client, arguments["emoji_id"])
    name = _name(arguments, "edit_application_emoji", required=False)

    action = "edit_application_emoji"
    targets = {"emoji_id": str(emoji.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"emojiId": str(emoji.id)})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs: Dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    updated = await emoji.edit(**kwargs)
    return json_text(
        {"status": "executed", "action": action, "emoji": _emoji_row(updated)}
    )


async def handle_delete_application_emoji(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_application_emoji")
    client = _client(deps, "delete_application_emoji")
    emoji = await _fetch_application_emoji(client, arguments["emoji_id"])
    application_id = getattr(client, "application_id", None)
    if application_id is None:
        raise ValueError(
            "delete_application_emoji: the current runtime does not expose an "
            f"application id, so application emoji '{emoji.id}' cannot be deleted"
        )
    delete = getattr(getattr(client, "http", None), "delete_application_emoji", None)
    if delete is None:
        raise ValueError(
            "delete_application_emoji: the current runtime exposes no delete path "
            "for application emoji (discord.Client has no delete_application_emoji "
            "method and http.delete_application_emoji is unavailable)"
        )

    action = "delete_application_emoji"
    targets = {"emoji_id": str(emoji.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"emojiId": str(emoji.id)})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await delete(application_id, emoji.id)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "emojiId": str(emoji.id),
        }
    )


async def handle_list_application_emojis(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "list_application_emojis")
    client = _client(deps, "list_application_emojis")
    emojis = await client.fetch_application_emojis()
    rows = [_emoji_row(emoji) for emoji in emojis]
    return json_text({"count": len(rows), "emojis": rows})


# --- stickers ---------------------------------------------------------------


async def handle_create_sticker(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_sticker")
    guild = await gateway.resolve_guild(arguments["server_id"])
    name = _name(arguments, "create_sticker")
    description = str(arguments["description"])
    sticker_emoji = str(arguments["emoji"])
    file_path = _sticker_file_path(arguments["file_path"])
    reason = require_reason(arguments.get("reason"), "create_sticker")

    action = "create_sticker"
    targets = {
        "server_id": str(guild.id),
        "name": name,
        "file_path": file_path,
    }
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action, targets, {"serverId": str(guild.id), "name": name}
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    try:
        file = discord.File(file_path)
    except OSError as exc:
        raise ValueError(f"file_path: cannot read '{file_path}': {exc}") from exc
    sticker = await guild.create_sticker(
        name=name,
        description=description,
        emoji=sticker_emoji,
        file=file,
        reason=reason,
    )
    return json_text(
        {"status": "executed", "action": action, "sticker": _sticker_row(sticker)}
    )


async def handle_edit_sticker(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_sticker")
    guild = await gateway.resolve_guild(arguments["server_id"])
    sticker = await _resolve_sticker(guild, arguments["sticker_id"])
    name = _name(arguments, "edit_sticker", required=False)
    description = arguments.get("description")
    sticker_emoji = arguments.get("emoji")
    reason = arguments.get("reason")

    # edit() uses the MISSING sentinel: only build kwargs for supplied fields
    fields: Dict[str, Any] = {}
    if name is not None:
        fields["name"] = name
    if description is not None:
        fields["description"] = str(description)
    if sticker_emoji is not None:
        fields["emoji"] = str(sticker_emoji)

    action = "edit_sticker"
    targets = {"server_id": str(guild.id), "sticker_id": str(sticker.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {
                    "serverId": str(guild.id),
                    "stickerId": str(sticker.id),
                    "fields": fields,
                },
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs = dict(fields)
    if reason is not None:
        kwargs["reason"] = reason
    updated = await sticker.edit(**kwargs)
    return json_text(
        {"status": "executed", "action": action, "sticker": _sticker_row(updated)}
    )


async def handle_delete_sticker(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_sticker")
    guild = await gateway.resolve_guild(arguments["server_id"])
    sticker = await _resolve_sticker(guild, arguments["sticker_id"])
    reason = require_reason(arguments.get("reason"), "delete_sticker")

    action = "delete_sticker"
    targets = {"server_id": str(guild.id), "sticker_id": str(sticker.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"stickerId": str(sticker.id)})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await guild.delete_sticker(sticker, reason=reason)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "serverId": str(guild.id),
            "stickerId": str(sticker.id),
        }
    )


async def handle_list_stickers(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "list_stickers")
    guild = await gateway.resolve_guild(arguments["server_id"])
    stickers = await guild.fetch_stickers()
    rows = [_sticker_row(sticker) for sticker in stickers]
    return json_text(
        {"serverId": str(guild.id), "count": len(rows), "stickers": rows}
    )


# --- soundboard ------------------------------------------------------------


async def handle_create_soundboard_sound(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_soundboard_sound")
    guild = await gateway.resolve_guild(arguments["server_id"])
    name = _name(arguments, "create_soundboard_sound")
    volume = _volume(arguments)
    emoji_value = parse_emoji(guild, arguments.get("emoji"))
    reason = arguments.get("reason")
    sound_bytes = await _download("sound_url", arguments["sound_url"])

    action = "create_soundboard_sound"
    targets = {"server_id": str(guild.id), "name": name}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action, targets, {"serverId": str(guild.id), "name": name}
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs: Dict[str, Any] = {"name": name, "sound": sound_bytes}
    if volume is not None:
        kwargs["volume"] = volume
    if emoji_value is not None:
        kwargs["emoji"] = emoji_value
    if reason is not None:
        kwargs["reason"] = reason
    sound = await guild.create_soundboard_sound(**kwargs)
    return json_text(
        {"status": "executed", "action": action, "sound": _sound_row(sound)}
    )


async def handle_list_soundboard_sounds(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "list_soundboard_sounds")
    guild = await gateway.resolve_guild(arguments["server_id"])
    sounds = await guild.fetch_soundboard_sounds()
    rows = [_sound_row(sound) for sound in sounds]
    return json_text(
        {"serverId": str(guild.id), "count": len(rows), "sounds": rows}
    )


async def handle_edit_soundboard_sound(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_soundboard_sound")
    guild = await gateway.resolve_guild(arguments["server_id"])
    sound = await _resolve_sound(guild, arguments["sound_id"])
    name = _name(arguments, "edit_soundboard_sound", required=False)
    volume = _volume(arguments)
    emoji_value = parse_emoji(guild, arguments.get("emoji"))
    reason = arguments.get("reason")

    action = "edit_soundboard_sound"
    targets = {"server_id": str(guild.id), "sound_id": str(sound.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"soundId": str(sound.id)})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs: Dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if volume is not None:
        kwargs["volume"] = volume
    if emoji_value is not None:
        kwargs["emoji"] = emoji_value
    if reason is not None:
        kwargs["reason"] = reason
    updated = await sound.edit(**kwargs)
    return json_text(
        {"status": "executed", "action": action, "sound": _sound_row(updated)}
    )


async def handle_delete_soundboard_sound(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_soundboard_sound")
    guild = await gateway.resolve_guild(arguments["server_id"])
    sound = await _resolve_sound(guild, arguments["sound_id"])
    reason = require_reason(arguments.get("reason"), "delete_soundboard_sound")

    action = "delete_soundboard_sound"
    targets = {"server_id": str(guild.id), "sound_id": str(sound.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"soundId": str(sound.id)})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await sound.delete(reason=reason)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "serverId": str(guild.id),
            "soundId": str(sound.id),
        }
    )


async def handle_send_soundboard_sound(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "send_soundboard_sound")
    guild = await gateway.resolve_guild(arguments["server_id"])
    channel = await _resolve_channel(guild, arguments["channel_id"])
    if not hasattr(channel, "send_sound"):
        raise ValueError(
            f"Channel '{arguments['channel_id']}' in server '{guild.name}' has type "
            f"'{getattr(channel, 'type', 'unknown')}'; send_soundboard_sound "
            "requires a voice channel"
        )
    sound = await _resolve_sound(guild, arguments["sound_id"])
    sound_guild = getattr(sound, "guild", None)
    if sound_guild is not None and getattr(sound_guild, "id", None) != guild.id:
        raise ValueError(
            f"Sound '{arguments['sound_id']}' belongs to a different server than "
            f"'{guild.name}'"
        )
    await channel.send_sound(sound)
    return json_text(
        {
            "status": "executed",
            "action": "send_soundboard_sound",
            "serverId": str(guild.id),
            "channelId": str(channel.id),
            "soundId": str(sound.id),
        }
    )
