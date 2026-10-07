import datetime
import json
from typing import Any, Dict, List

import discord
from mcp.types import TextContent

from discord_mcp.core.images import load_image_bytes
from discord_mcp.core.resolve import try_int
from discord_mcp.core.safety import build_dry_run_result


async def handle_get_server_info(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    # Fetch fresh instead of reading the cached guild object: the gateway cache is
    # built from GUILD_CREATE, which omits fields for large guilds, so cached reads
    # can report a stale/empty description and verification level.
    guild = await gateway.fetch_guild(arguments["server_id"])
    info = {
        "name": guild.name,
        "id": str(guild.id),
        "owner_id": str(guild.owner_id),
        "member_count": guild.member_count
        or getattr(guild, "approximate_member_count", None),
        "created_at": guild.created_at.isoformat(),
        "description": guild.description,
        "verification_level": str(guild.verification_level),
        "premium_tier": guild.premium_tier,
        "explicit_content_filter": str(guild.explicit_content_filter),
    }
    return [
        TextContent(
            type="text",
            text="Server Information:\n"
            + "\n".join(f"{k}: {v}" for k, v in info.items()),
        )
    ]


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


async def handle_list_members(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    limit = min(int(arguments.get("limit", 100)), 1000)

    members = []
    async for member in guild.fetch_members(limit=limit):
        members.append(
            {
                "id": str(member.id),
                "name": member.name,
                "nick": member.nick,
                "joined_at": member.joined_at.isoformat() if member.joined_at else None,
                "roles": [str(role.id) for role in member.roles[1:]],
            }
        )

    return [
        TextContent(
            type="text",
            text=f"Server Members ({len(members)}):\n"
            + "\n".join(
                f"{m['name']} (ID: {m['id']}, Roles: {', '.join(m['roles'])})"
                for m in members
            ),
        )
    ]


async def handle_list_servers(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    servers = [
        {
            "id": str(guild.id),
            "name": guild.name,
            "member_count": guild.member_count,
            "created_at": guild.created_at.isoformat(),
        }
        for guild in gateway.client.guilds
    ]

    return [
        TextContent(
            type="text",
            text=f"Available Servers ({len(servers)}):\n"
            + "\n".join(
                f"{s['name']} (ID: {s['id']}, Members: {s['member_count']})"
                for s in servers
            ),
        )
    ]


_VERIFICATION_LEVELS = {
    "0": 0,
    "none": 0,
    "1": 1,
    "low": 1,
    "2": 2,
    "medium": 2,
    "3": 3,
    "high": 3,
    "4": 4,
    "highest": 4,
    "very_high": 4,
}

_EXPLICIT_CONTENT_FILTERS = {
    "0": 0,
    "disabled": 0,
    "1": 1,
    "no_role": 1,
    "members_without_roles": 1,
    "2": 2,
    "all_members": 2,
}


def _coerce_enum(mapping: Dict[str, int], value: Any, label: str) -> int:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in mapping:
        raise ValueError(
            f"unknown {label} '{value}'; expected one of {', '.join(sorted(mapping))}"
        )
    return mapping[key]


_NOTIFICATION_LEVELS = {
    "0": 0,
    "all_messages": 0,
    "all": 0,
    "1": 1,
    "only_mentions": 1,
    "mentions": 1,
}

_MFA_LEVELS = {"0": 0, "none": 0, "1": 1, "elevated": 1}

_AFK_TIMEOUTS = (60, 300, 900, 1800, 3600)

_IMAGE_FIELDS = ("icon", "banner", "splash", "discovery_splash")
_CHANNEL_FIELDS = (
    "afk_channel",
    "system_channel",
    "rules_channel",
    "public_updates_channel",
    "safety_alerts_channel",
    "widget_channel",
)
_BOOL_FIELDS = (
    "community",
    "discoverable",
    "invites_disabled",
    "widget_enabled",
    "premium_progress_bar_enabled",
    "raid_alerts_disabled",
)
_DATETIME_FIELDS = ("invites_disabled_until", "dms_disabled_until")
_STRING_FIELDS = ("name", "description", "preferred_locale", "vanity_code")
_ENUM_FIELDS = (
    "verification_level",
    "explicit_content_filter",
    "default_notifications",
    "mfa_level",
)

SUPPORTED_UPDATE_GUILD_FIELDS = (
    "server_id",
    *_STRING_FIELDS,
    *_ENUM_FIELDS,
    *_BOOL_FIELDS,
    *_CHANNEL_FIELDS,
    "afk_timeout",
    "system_channel_flags",
    "owner",
    *_IMAGE_FIELDS,
    *_DATETIME_FIELDS,
    "reason",
)


def _resolve_channel(guild: Any, value: Any, where: str):
    if value is None:
        return None
    parsed = try_int(value)
    if parsed is not None:
        channel = guild.get_channel(parsed)
        if channel is None:
            raise ValueError(f"{where}: channel '{value}' not found")
        return channel
    wanted = str(value).strip().lower().removeprefix("#")
    matches = [
        channel
        for channel in getattr(guild, "channels", []) or []
        if str(getattr(channel, "name", "")).strip().lower() == wanted
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"{where}: channel name '{value}' is ambiguous; use the channel id"
        )
    raise ValueError(f"{where}: channel '{value}' not found")


def _system_channel_flags(value: Any) -> Any:
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return discord.SystemChannelFlags._from_value(int(value))
    if isinstance(value, (list, tuple, set)):
        flags = discord.SystemChannelFlags()
        for name in value:
            key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
            if key not in discord.SystemChannelFlags.VALID_FLAGS:
                raise ValueError(
                    f"system_channel_flags: unknown flag '{name}'; expected one of "
                    f"{', '.join(sorted(discord.SystemChannelFlags.VALID_FLAGS))}"
                )
            setattr(flags, key, True)
        return flags
    parsed = try_int(value)
    if parsed is None:
        raise ValueError(
            "system_channel_flags must be an int bitfield or an array of flag names"
        )
    return discord.SystemChannelFlags._from_value(parsed)


def _parse_datetime(value: Any, where: str) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{where} must be an ISO8601 timestamp or null, got {value!r}")


async def _build_guild_updates(guild: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
    unsupported = sorted(
        key for key in arguments if key not in SUPPORTED_UPDATE_GUILD_FIELDS
    )
    if unsupported:
        raise ValueError(
            f"unsupported_fields: {', '.join(unsupported)}. Supported: "
            f"{', '.join(sorted(SUPPORTED_UPDATE_GUILD_FIELDS))}"
        )

    updates: Dict[str, Any] = {}
    for field in _STRING_FIELDS:
        if field in arguments:
            value = arguments[field]
            updates[field] = None if value is None else str(value)
    for field in _BOOL_FIELDS:
        if field in arguments and arguments[field] is not None:
            updates[field] = bool(arguments[field])
    for field in _CHANNEL_FIELDS:
        if field in arguments:
            updates[field] = _resolve_channel(guild, arguments[field], field)
    for field in _DATETIME_FIELDS:
        if field in arguments:
            updates[field] = _parse_datetime(arguments[field], field)
    for field in _IMAGE_FIELDS:
        if field in arguments:
            updates[field] = await load_image_bytes(arguments[field], field)

    if arguments.get("verification_level") is not None:
        updates["verification_level"] = discord.VerificationLevel(
            _coerce_enum(
                _VERIFICATION_LEVELS,
                arguments["verification_level"],
                "verification_level",
            )
        )
    if arguments.get("explicit_content_filter") is not None:
        updates["explicit_content_filter"] = discord.ContentFilter(
            _coerce_enum(
                _EXPLICIT_CONTENT_FILTERS,
                arguments["explicit_content_filter"],
                "explicit_content_filter",
            )
        )
    if arguments.get("default_notifications") is not None:
        updates["default_notifications"] = discord.NotificationLevel(
            _coerce_enum(
                _NOTIFICATION_LEVELS,
                arguments["default_notifications"],
                "default_notifications",
            )
        )
    if arguments.get("mfa_level") is not None:
        updates["mfa_level"] = discord.MFALevel(
            _coerce_enum(_MFA_LEVELS, arguments["mfa_level"], "mfa_level")
        )
    if arguments.get("afk_timeout") is not None:
        timeout = int(arguments["afk_timeout"])
        if timeout not in _AFK_TIMEOUTS:
            raise ValueError(
                f"afk_timeout must be one of {', '.join(str(v) for v in _AFK_TIMEOUTS)}"
            )
        updates["afk_timeout"] = timeout
    if "system_channel_flags" in arguments:
        updates["system_channel_flags"] = _system_channel_flags(
            arguments["system_channel_flags"]
        )
    if arguments.get("owner") is not None:
        owner_id = try_int(arguments["owner"])
        if owner_id is None:
            raise ValueError(f"owner must be a user id, got {arguments['owner']!r}")
        updates["owner"] = discord.Object(id=owner_id)
    return updates


async def handle_update_guild(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    updates = await _build_guild_updates(guild, arguments)
    if not updates:
        raise ValueError(
            "nothing to update: pass at least one of "
            + ", ".join(sorted(SUPPORTED_UPDATE_GUILD_FIELDS))
        )

    reason = arguments.get("reason")
    try:
        await guild.edit(reason=reason, **updates)
    except discord.Forbidden as exc:
        raise ValueError(
            f"Cannot edit server '{guild.id}': {exc}. MANAGE_GUILD is required, and adding or "
            "removing the COMMUNITY feature needs ADMINISTRATOR. Image fields (banner, splash, "
            "discovery_splash, animated icon) also need the matching guild feature."
        )
    except discord.HTTPException as exc:
        raise ValueError(f"Cannot edit server '{guild.id}': {exc}")

    payload = {
        "status": "applied",
        "server_id": str(guild.id),
        "applied_fields": sorted(key for key in updates if key != "reason"),
        "name": guild.name,
        "description": guild.description,
        "preferred_locale": str(getattr(guild, "preferred_locale", None)),
        "verification_level": str(guild.verification_level),
        "explicit_content_filter": str(guild.explicit_content_filter),
        "default_notifications": str(getattr(guild, "default_notifications", None)),
        "features": sorted(getattr(guild, "features", []) or []),
        "reason": reason,
    }
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
