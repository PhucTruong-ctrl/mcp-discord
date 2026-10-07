import asyncio
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import discord
from discord.channel import VocalGuildChannel
from mcp.types import TextContent

from discord_mcp.core.common import json_text, member_row, require_gateway, user_row
from discord_mcp.core.permissions import permission_names
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.validation import require_reason, validate_snowflake
from discord_mcp.tools.handlers.member_admin import _nickname_value

_URL_SCHEMES = ("http", "https")


def _is_dry_run(arguments: Dict[str, Any]) -> bool:
    return bool(arguments.get("dry_run", True))


def _optional_reason(arguments: Dict[str, Any]) -> Optional[str]:
    reason = str(arguments.get("reason", "") or "").strip()
    return reason or None


def _opt_bool(arguments: Dict[str, Any], key: str) -> Optional[bool]:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean (got {value!r})")
    return value


async def _resolve_member(guild: Any, member_id: str) -> Any:
    member_id_int = validate_snowflake(str(member_id))
    member = guild.get_member(member_id_int)
    if member is not None:
        return member
    try:
        return await guild.fetch_member(member_id_int)
    except discord.NotFound:
        raise ValueError(
            f"Member '{member_id}' not found in server '{guild.name}'"
        ) from None


def _resolve_voice_channel(guild: Any, channel_id: Any) -> Any:
    channel_id_int = validate_snowflake(str(channel_id))
    channel = guild.get_channel(channel_id_int)
    if channel is None:
        for candidate in getattr(guild, "channels", []):
            if getattr(candidate, "id", None) == channel_id_int:
                channel = candidate
                break
    if channel is None:
        raise ValueError(f"Channel '{channel_id}' not found in server '{guild.name}'")
    if not isinstance(channel, VocalGuildChannel):
        channel_type = getattr(channel, "type", None)
        type_name = getattr(channel_type, "name", str(channel_type))
        raise ValueError(
            f"Channel '{channel_id}' in server '{guild.name}' is a {type_name} "
            "channel, not a voice channel"
        )
    return channel


def _download_image(url: Any, field: str) -> bytes:
    """Read an http(s) URL in-process; reject every other scheme outright."""
    if not isinstance(url, str):
        raise ValueError(f"{field} must be an http(s) URL string or null (got {url!r})")
    scheme = urlparse(url).scheme.lower()
    if scheme not in _URL_SCHEMES:
        raise ValueError(f"{field}: only http/https URLs are supported, got '{url}'")
    try:
        with urlopen(url, timeout=30) as response:  # noqa: S310 - scheme checked above
            return response.read()
    except (URLError, OSError) as exc:
        raise ValueError(f"{field}: failed to download '{url}': {exc}") from exc


async def _image_bytes(url: Any, field: str) -> bytes:
    return await asyncio.to_thread(_download_image, url, field)


def _display_icon_url(value: Any) -> Optional[str]:
    if value is None:
        return None
    url = getattr(value, "url", None)
    return str(url) if url is not None else str(value)


def _tags_payload(tags: Any) -> Optional[Dict[str, Any]]:
    if tags is None:
        return None
    bot_id = getattr(tags, "bot_id", None)
    integration_id = getattr(tags, "integration_id", None)
    listing_id = getattr(tags, "subscription_listing_id", None)
    return {
        "botId": str(bot_id) if bot_id is not None else None,
        "integrationId": str(integration_id) if integration_id is not None else None,
        "subscriptionListingId": str(listing_id) if listing_id is not None else None,
        "premiumSubscriber": bool(tags.is_premium_subscriber()),
        "availableForPurchase": bool(tags.is_available_for_purchase()),
        "guildConnections": bool(tags.is_guild_connection()),
    }


def _role_detail_row(role: Any, counts_by_id: Dict[str, int]) -> Dict[str, Any]:
    tags = getattr(role, "tags", None)
    return {
        "id": str(role.id),
        "name": getattr(role, "name", None),
        "position": int(getattr(role, "position", 0)),
        "hoist": bool(getattr(role, "hoist", False)),
        "mentionable": bool(getattr(role, "mentionable", False)),
        "managed": bool(getattr(role, "managed", False)),
        "botManaged": bool(role.is_bot_managed()),
        "integration": bool(role.is_integration()),
        "premiumSubscriber": (
            None if tags is None else bool(tags.is_premium_subscriber())
        ),
        "displayIconUrl": _display_icon_url(getattr(role, "display_icon", None)),
        "tags": _tags_payload(tags),
        "memberCount": counts_by_id.get(str(role.id)),
        "permissionNames": permission_names(getattr(role, "permissions", None)),
    }


async def handle_change_member_voice_state(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "change_member_voice_state")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))
    member_id = str(arguments["member_id"])
    member = await _resolve_member(guild, member_id)

    channel_id = arguments.get("channel_id")
    channel = (
        _resolve_voice_channel(guild, channel_id)
        if channel_id not in (None, "")
        else None
    )
    mute = _opt_bool(arguments, "mute")
    deafen = _opt_bool(arguments, "deafen")
    reason = _optional_reason(arguments)

    # discord.py 2.7.1: Guild.change_voice_state is gateway opcode 4 and can only
    # address the bot's own session. A member's voice state (move / server mute /
    # server deafen) is the Edit Guild Member payload: Member.edit(voice_channel=,
    # mute=, deafen=).
    action = "change_member_voice_state"
    targets = {"member_id": str(member.id), "server_id": str(guild.id)}
    details = {
        "channelId": str(channel.id) if channel is not None else None,
        "mute": mute,
        "deafen": deafen,
    }
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs: Dict[str, Any] = {"voice_channel": channel}  # None disconnects
    if mute is not None:
        kwargs["mute"] = mute
    if deafen is not None:
        kwargs["deafen"] = deafen
    if reason is not None:
        kwargs["reason"] = reason
    await member.edit(**kwargs)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "serverId": targets["server_id"],
            "memberId": targets["member_id"],
            **details,
        }
    )


async def handle_move_member_voice(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "move_member_voice")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))
    member_id = str(arguments["member_id"])
    member = await _resolve_member(guild, member_id)

    channel_id = arguments.get("channel_id")
    channel = (
        _resolve_voice_channel(guild, channel_id)
        if channel_id not in (None, "")
        else None
    )
    reason = _optional_reason(arguments)

    action = "move_member_voice"
    targets = {"member_id": str(member.id), "server_id": str(guild.id)}
    details = {
        "channelId": str(channel.id) if channel is not None else None,
        "disconnected": channel is None,
    }
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await member.move_to(channel, reason=reason)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "serverId": targets["server_id"],
            "memberId": targets["member_id"],
            **details,
        }
    )


async def handle_request_to_speak(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "request_to_speak")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))
    member_id = str(arguments["member_id"])
    member = await _resolve_member(guild, member_id)
    reason = require_reason(arguments.get("reason"), "request_to_speak")

    voice = getattr(member, "voice", None)
    if voice is None or getattr(voice, "channel", None) is None:
        raise ValueError(
            f"Member '{member_id}' in server '{guild.name}' is not connected "
            "to a voice channel"
        )

    action = "request_to_speak"
    targets = {"member_id": str(member.id), "server_id": str(guild.id)}
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, {"reason": reason}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await member.request_to_speak()
    return json_text(
        {
            "status": "executed",
            "action": action,
            "serverId": targets["server_id"],
            "memberId": targets["member_id"],
        }
    )


async def handle_get_member_voice_state(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_member_voice_state")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))
    member_id = str(arguments["member_id"])
    member = await _resolve_member(guild, member_id)

    state = getattr(member, "voice", None)
    if state is None:
        try:
            state = await member.fetch_voice()
        except discord.NotFound:
            state = None

    channel = getattr(state, "channel", None) if state is not None else None
    in_voice = state is not None and channel is not None
    payload = {
        "serverId": str(guild.id),
        "memberId": str(member.id),
        "inVoice": in_voice,
        "channelId": str(channel.id) if channel is not None else None,
        "channelName": getattr(channel, "name", None) if channel is not None else None,
        "muted": bool(state.mute) if state is not None else None,
        "deafen": bool(state.deaf) if state is not None else None,
        "selfMute": bool(state.self_mute) if state is not None else None,
        "selfDeaf": bool(state.self_deaf) if state is not None else None,
        "streaming": bool(state.self_stream) if state is not None else None,
        "video": bool(state.self_video) if state is not None else None,
    }
    return json_text(payload)


async def handle_edit_member_profile(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_member_profile")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))
    member_id = str(arguments["member_id"])
    member = await _resolve_member(guild, member_id)

    has_nickname = "nickname" in arguments
    has_avatar = "avatar_url" in arguments
    has_banner = "banner_url" in arguments
    has_bio = "bio" in arguments
    if not (has_nickname or has_avatar or has_banner or has_bio):
        raise ValueError(
            "edit_member_profile requires at least one of: "
            "nickname, avatar_url, banner_url, bio"
        )

    kwargs: Dict[str, Any] = {}
    if has_nickname:
        kwargs["nick"] = _nickname_value(arguments.get("nickname"))
    if has_avatar:
        avatar_url = arguments.get("avatar_url")
        kwargs["avatar"] = (
            await _image_bytes(avatar_url, "avatar_url")
            if avatar_url is not None
            else None
        )
    if has_banner:
        banner_url = arguments.get("banner_url")
        kwargs["banner"] = (
            await _image_bytes(banner_url, "banner_url")
            if banner_url is not None
            else None
        )
    if has_bio:
        bio = arguments.get("bio")
        if bio is not None and not isinstance(bio, str):
            raise ValueError(f"bio must be a string or null (got {bio!r})")
        kwargs["bio"] = bio
    reason = _optional_reason(arguments)

    action = "edit_member_profile"
    targets = {"member_id": str(member.id), "server_id": str(guild.id)}
    details = {
        key: arguments.get(field)
        for key, field in (
            ("nickname", "nickname"),
            ("avatarUrl", "avatar_url"),
            ("bannerUrl", "banner_url"),
            ("bio", "bio"),
        )
        if field in arguments
    }
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    if reason is not None:
        kwargs["reason"] = reason
    updated = await member.edit(**kwargs)
    row = updated if updated is not None else member
    return json_text(
        {
            "status": "executed",
            "action": action,
            "serverId": targets["server_id"],
            "memberId": targets["member_id"],
            "member": member_row(row),
        }
    )


async def handle_create_dm_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_dm_channel")
    client = deps.get("discord_client")
    if not client:
        raise ValueError("discord_client is required for create_dm_channel")

    user_id = str(arguments["user_id"])
    user_id_int = validate_snowflake(user_id)
    try:
        user = await client.fetch_user(user_id_int)
    except discord.NotFound:
        raise ValueError(f"User '{user_id}' not found") from None

    channel = await user.create_dm()
    return json_text(
        {
            "status": "executed",
            "action": "create_dm_channel",
            "channelId": str(channel.id),
            "recipient": user_row(user),
        }
    )


async def handle_update_bot_profile(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "update_bot_profile")
    client = deps.get("discord_client")
    if not client:
        raise ValueError("discord_client is required for update_bot_profile")
    user = getattr(client, "user", None)
    if user is None:
        raise ValueError("discord_client.user is required for update_bot_profile")

    has_username = "username" in arguments
    has_avatar = "avatar_url" in arguments
    has_banner = "banner_url" in arguments
    if not (has_username or has_avatar or has_banner):
        raise ValueError(
            "update_bot_profile requires at least one of: "
            "username, avatar_url, banner_url"
        )

    kwargs: Dict[str, Any] = {}
    if has_username:
        username = arguments.get("username")
        if not isinstance(username, str) or not username.strip():
            raise ValueError(f"username must be a non-empty string (got {username!r})")
        kwargs["username"] = username
    if has_avatar:
        avatar_url = arguments.get("avatar_url")
        kwargs["avatar"] = (
            await _image_bytes(avatar_url, "avatar_url")
            if avatar_url is not None
            else None
        )
    if has_banner:
        banner_url = arguments.get("banner_url")
        kwargs["banner"] = (
            await _image_bytes(banner_url, "banner_url")
            if banner_url is not None
            else None
        )

    action = "update_bot_profile"
    targets = {"user_id": str(user.id)}
    details = {
        key: arguments.get(field)
        for key, field in (
            ("username", "username"),
            ("avatarUrl", "avatar_url"),
            ("bannerUrl", "banner_url"),
        )
        if field in arguments
    }
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    updated = await user.edit(**kwargs)
    row = updated if updated is not None else user
    return json_text({"status": "executed", "action": action, "user": user_row(row)})


async def handle_set_role_icon(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "set_role_icon")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))
    role_id = str(arguments["role_id"])
    role = guild.get_role(validate_snowflake(role_id))
    if role is None:
        raise ValueError(f"Role '{role_id}' not found in server '{guild.name}'")

    icon_url = arguments.get("icon_url")
    icon = await _image_bytes(icon_url, "icon_url") if icon_url is not None else None
    reason = _optional_reason(arguments)

    action = "set_role_icon"
    targets = {"role_id": str(role.id), "server_id": str(guild.id)}
    details = {"iconUrl": icon_url, "clearing": icon is None}
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    updated = await role.edit(display_icon=icon, reason=reason)
    edited = updated if updated is not None else role
    return json_text(
        {
            "status": "executed",
            "action": action,
            "serverId": targets["server_id"],
            "roleId": targets["role_id"],
            "displayIconUrl": _display_icon_url(getattr(edited, "display_icon", None)),
        }
    )


async def handle_reorder_roles(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "reorder_roles")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))
    raw_positions = arguments.get("positions")
    if not isinstance(raw_positions, dict) or not raw_positions:
        raise ValueError(
            "positions must be a non-empty object mapping role IDs to "
            "integer positions"
        )
    reason = require_reason(arguments.get("reason"), "reorder_roles")

    # discord.py's edit_role_positions dereferences .id on every key, so keys
    # must be resolved Role objects, not raw ints.
    resolved: Dict[str, Any] = {}
    for key, value in raw_positions.items():
        try:
            role_id_int = int(str(key))
        except (TypeError, ValueError):
            raise ValueError(
                f"positions: '{key}' is not a valid role id "
                "(expected a numeric snowflake)"
            ) from None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"positions['{key}'] must be an integer position (got {value!r})"
            )
        role = guild.get_role(role_id_int)
        if role is None:
            raise ValueError(f"Role '{key}' not found in server '{guild.name}'")
        resolved[str(role_id_int)] = (role, int(value))

    action = "reorder_roles"
    positions = {key: entry[1] for key, entry in resolved.items()}
    targets = {"positions": positions, "server_id": str(guild.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"count": len(resolved)})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await guild.edit_role_positions(
        {role: position for role, position in resolved.values()}, reason=reason
    )
    return json_text(
        {
            "status": "executed",
            "action": action,
            "serverId": targets["server_id"],
            "positions": positions,
            "count": len(resolved),
        }
    )


async def handle_get_role_details(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_role_details")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))

    role_id = arguments.get("role_id")
    if role_id:
        role = guild.get_role(validate_snowflake(str(role_id)))
        if role is None:
            raise ValueError(f"Role '{role_id}' not found in server '{guild.name}'")
        roles = [role]
    else:
        roles = sorted(
            guild.roles, key=lambda entry: (int(entry.position), int(entry.id)),
            reverse=True,
        )

    counts = await guild.role_member_counts()
    counts_by_id = {
        str(getattr(key, "id", key)): count for key, count in counts.items()
    }
    rows = [_role_detail_row(role, counts_by_id) for role in roles]
    return json_text(
        {"serverId": str(guild.id), "count": len(rows), "roles": rows}
    )
