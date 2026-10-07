"""Guild scheduled-event and stage-instance handlers."""

import asyncio
import datetime
from typing import Any, Dict, List
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import discord
from mcp.types import TextContent

from discord_mcp.core.common import as_id, json_text, require_gateway, user_row
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.validation import (
    require_reason,
    validate_enum,
    validate_limit,
    validate_snowflake,
)

# MCP-facing entity names; discord.py's enum member for voice is `voice`.
_ENTITY_TYPES = {
    "stage_instance": discord.EntityType.stage_instance,
    "voice_channel": discord.EntityType.voice,
    "external": discord.EntityType.external,
}
_ENTITY_TYPE_TO_NAME = {
    discord.EntityType.stage_instance: "stage_instance",
    discord.EntityType.voice: "voice_channel",
    discord.EntityType.external: "external",
}
_EDITABLE_EVENT_FIELDS = (
    "name, description, channel_id, start_time, end_time, privacy_level, "
    "entity_type, image_url, location"
)


def _is_dry_run(arguments: Dict[str, Any]) -> bool:
    return bool(arguments.get("dry_run", True))


def _channel_type(channel: Any) -> str:
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
    return _channel_type(channel) == expected


def _resolve_channel(guild: Any, channel_id: Any, server: str) -> Any:
    channel = None
    if hasattr(guild, "get_channel"):
        try:
            channel = guild.get_channel(int(channel_id))
        except (TypeError, ValueError):
            channel = None
    if channel is None:
        for candidate in getattr(guild, "channels", []):
            if str(getattr(candidate, "id", "")) == str(channel_id):
                channel = candidate
                break
    if channel is None:
        raise ValueError(f"Channel '{channel_id}' not found in server '{server}'")
    return channel


def _resolve_stage_channel(guild: Any, channel_id: Any, server: str) -> Any:
    channel = _resolve_channel(guild, channel_id, server)
    if not _matches_channel_type(channel, "stage"):
        raise ValueError(
            f"Channel '{channel_id}' in server '{server}' is a "
            f"{_channel_type(channel)} channel, not a stage channel"
        )
    return channel


def _present(arguments: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    """Kwargs the caller actually set — discord.py treats ``None`` as a real value."""
    return {key: arguments[key] for key in keys if arguments.get(key) is not None}


def _parse_time(value: Any, field: str) -> datetime.datetime:
    text = str(value).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"{field} must be an ISO-8601 timestamp such as "
            f"2026-12-01T18:00:00+00:00 (got {value!r})"
        ) from None
    if parsed.tzinfo is None:
        raise ValueError(
            f"{field} must include a timezone offset such as +00:00 or Z "
            f"(got {value!r})"
        )
    return parsed


def _parse_entity_type(value: Any) -> discord.EntityType:
    normalized = validate_enum(str(value), list(_ENTITY_TYPES), "entity_type")
    return _ENTITY_TYPES[normalized]


def _parse_privacy_level(value: Any) -> discord.PrivacyLevel | None:
    if value is None:
        return None
    validate_enum(str(value), ["guild_only"], "privacy_level")
    return discord.PrivacyLevel.guild_only


def _validate_image_url(value: Any) -> str:
    url = str(value).strip()
    if urlparse(url).scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"image_url: only http/https URLs are supported, got '{url}'"
        )
    return url


def _download_image(url: str) -> bytes:
    try:
        with urlopen(url, timeout=30) as response:  # noqa: S310 - scheme checked above
            return response.read()
    except (URLError, OSError) as exc:
        raise ValueError(f"image_url: failed to download '{url}': {exc}") from exc


async def _fetch_image(url: str) -> bytes:
    return await asyncio.to_thread(_download_image, url)


async def _fetch_event(guild: Any, event_id: Any, server: str, *, with_counts=False):
    event_id_int = validate_snowflake(event_id)
    try:
        return await guild.fetch_scheduled_event(
            event_id_int, with_counts=with_counts
        )
    except discord.NotFound:
        raise ValueError(
            f"Scheduled event '{event_id}' not found in server '{server}'"
        ) from None


async def _fetch_instance(channel: Any, server: str) -> Any:
    try:
        return await channel.fetch_instance()
    except discord.NotFound:
        raise ValueError(
            f"No stage instance is running on channel '{channel.id}' "
            f"in server '{server}'"
        ) from None


def _enum_name(value: Any) -> Any:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        # discord.py fills ScheduledEvent.privacy_level from the event *status*
        # (scheduled_event.py `_update`), so non-guild_only events arrive as
        # `unknown_<status>` proxies with no meaningful privacy name.
        return None if name.startswith("unknown_") else name
    return None if value is None else str(value)


def _entity_type_name(event: Any) -> Any:
    name = _ENTITY_TYPE_TO_NAME.get(event.entity_type)
    return name or _enum_name(event.entity_type)


def _event_row(event: Any, with_user_count: bool) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": str(event.id),
        "name": event.name,
        "description": event.description,
        "entityType": _entity_type_name(event),
        "entityId": as_id(event.entity_id),
        "entityMetadata": (
            {"location": event.location} if event.location is not None else None
        ),
        "status": _enum_name(event.status),
        "startTime": (
            event.start_time.isoformat() if event.start_time is not None else None
        ),
        "endTime": event.end_time.isoformat() if event.end_time is not None else None,
        "privacyLevel": _enum_name(event.privacy_level),
        "location": event.location,
    }
    if with_user_count:
        row["userCount"] = event.user_count
    row["creatorId"] = as_id(event.creator_id)
    row["channelId"] = as_id(event.channel_id)
    return row


def _stage_row(instance: Any, channel: Any) -> Dict[str, Any]:
    return {
        "channelId": str(instance.channel_id),
        "channelName": getattr(channel, "name", None),
        "topic": instance.topic,
        "privacyLevel": _enum_name(instance.privacy_level),
        "discoverableDisabled": bool(instance.discoverable_disabled),
        "guildScheduledEventId": as_id(instance.scheduled_event_id),
    }


async def handle_create_scheduled_event(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_scheduled_event")
    guild = await gateway.resolve_guild(arguments["server_id"])
    server = guild.name

    name = str(arguments["name"])
    entity_type = _parse_entity_type(arguments["entity_type"])
    entity_name = _ENTITY_TYPE_TO_NAME[entity_type]
    start_time = _parse_time(arguments["start_time"], "start_time")
    end_time = None
    if arguments.get("end_time") is not None:
        end_time = _parse_time(arguments["end_time"], "end_time")
    privacy_level = _parse_privacy_level(arguments.get("privacy_level"))
    location = arguments.get("location")
    description = arguments.get("description")
    channel_id = arguments.get("channel_id")
    image_url = None
    if arguments.get("image_url") is not None:
        image_url = _validate_image_url(arguments["image_url"])
    reason = arguments.get("reason")

    channel = None
    if entity_type in (discord.EntityType.stage_instance, discord.EntityType.voice):
        if channel_id is None:
            raise ValueError(
                f"channel_id is required when entity_type is '{entity_name}'"
            )
        if location is not None:
            raise ValueError(
                f"location must not be set when entity_type is '{entity_name}'"
            )
        channel = _resolve_channel(guild, channel_id, server)
        if entity_type is discord.EntityType.stage_instance:
            expected = "stage"
        else:
            expected = "voice"
        if not _matches_channel_type(channel, expected):
            raise ValueError(
                f"Channel '{channel_id}' in server '{server}' is a "
                f"{_channel_type(channel)} channel; entity_type '{entity_name}' "
                f"needs a {expected} channel"
            )
    else:
        if channel_id is not None:
            raise ValueError(
                "channel_id must not be set when entity_type is 'external'"
            )
        if location is None or not str(location).strip():
            raise ValueError("location is required when entity_type is 'external'")
        if end_time is None:
            raise ValueError("end_time is required when entity_type is 'external'")

    action = "create_scheduled_event"
    targets = {"server_id": str(arguments["server_id"]), "name": name}
    if _is_dry_run(arguments):
        details: Dict[str, Any] = {
            "name": name,
            "entityType": entity_name,
            "startTime": start_time.isoformat(),
        }
        if channel is not None:
            details["channelId"] = str(channel.id)
        if location is not None:
            details["location"] = location
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs: Dict[str, Any] = {
        "name": name,
        "start_time": start_time,
        "entity_type": entity_type,
    }
    if channel is not None:
        kwargs["channel"] = channel
    if end_time is not None:
        kwargs["end_time"] = end_time
    if privacy_level is not None:
        kwargs["privacy_level"] = privacy_level
    if description is not None:
        kwargs["description"] = description
    if location is not None:
        kwargs["location"] = location
    if image_url is not None:
        kwargs["image"] = await _fetch_image(image_url)
    if reason is not None:
        kwargs["reason"] = reason
    event = await guild.create_scheduled_event(**kwargs)
    return json_text(
        {"status": "executed", "action": action, "event": _event_row(event, False)}
    )


async def handle_get_scheduled_event(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_scheduled_event")
    guild = await gateway.resolve_guild(arguments["server_id"])
    with_user_count = bool(arguments.get("with_user_count", False))
    event = await _fetch_event(
        guild, arguments["event_id"], guild.name, with_counts=with_user_count
    )
    return json_text({"event": _event_row(event, with_user_count)})


async def handle_list_scheduled_events(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "list_scheduled_events")
    guild = await gateway.resolve_guild(arguments["server_id"])
    with_user_count = bool(arguments.get("with_user_count", False))
    events = await guild.fetch_scheduled_events(with_counts=with_user_count)
    rows = [
        _event_row(event, with_user_count)
        for event in sorted(events, key=lambda item: (item.start_time, item.id))
    ]
    return json_text(
        {"serverId": str(guild.id), "count": len(rows), "events": rows}
    )


async def handle_edit_scheduled_event(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_scheduled_event")
    guild = await gateway.resolve_guild(arguments["server_id"])
    server = guild.name

    updates = _present(arguments, "name", "description", "location")
    if arguments.get("start_time") is not None:
        updates["start_time"] = _parse_time(arguments["start_time"], "start_time")
    if arguments.get("end_time") is not None:
        updates["end_time"] = _parse_time(arguments["end_time"], "end_time")
    if arguments.get("privacy_level") is not None:
        updates["privacy_level"] = _parse_privacy_level(arguments["privacy_level"])
    if arguments.get("entity_type") is not None:
        updates["entity_type"] = _parse_entity_type(arguments["entity_type"])
    image_url = None
    if arguments.get("image_url") is not None:
        image_url = _validate_image_url(arguments["image_url"])
    channel_id = arguments.get("channel_id")
    if channel_id is not None:
        updates["channel"] = _resolve_channel(guild, channel_id, server)
    reason = arguments.get("reason")

    editable = sorted(updates)
    if image_url is not None:
        editable.append("image_url")
    if not editable:
        raise ValueError(
            "edit_scheduled_event requires at least one of: "
            + _EDITABLE_EVENT_FIELDS
        )

    event = await _fetch_event(guild, arguments["event_id"], server)
    action = "edit_scheduled_event"
    targets = {"server_id": str(arguments["server_id"]), "event_id": str(event.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {"eventId": str(event.id), "updates": editable},
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    if image_url is not None:
        updates["image"] = await _fetch_image(image_url)
    if reason is not None:
        updates["reason"] = reason
    edited = await event.edit(**updates)
    return json_text(
        {"status": "executed", "action": action, "event": _event_row(edited, False)}
    )


async def handle_delete_scheduled_event(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_scheduled_event")
    guild = await gateway.resolve_guild(arguments["server_id"])
    action = "delete_scheduled_event"
    reason = require_reason(arguments.get("reason"), action)
    event = await _fetch_event(guild, arguments["event_id"], guild.name)

    targets = {"server_id": str(arguments["server_id"]), "event_id": str(event.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {"eventId": str(event.id), "name": event.name},
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await event.delete(reason=reason)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "eventId": str(event.id),
            "name": event.name,
        }
    )


async def _transition(
    gateway: Any, arguments: Dict[str, Any], action: str, method: str
) -> List[TextContent]:
    guild = await gateway.resolve_guild(arguments["server_id"])
    event = await _fetch_event(guild, arguments["event_id"], guild.name)
    targets = {"server_id": str(arguments["server_id"]), "event_id": str(event.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {"eventId": str(event.id), "name": event.name},
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    updated = await getattr(event, method)(reason=arguments.get("reason"))
    return json_text(
        {"status": "executed", "action": action, "event": _event_row(updated, False)}
    )


async def handle_start_scheduled_event(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "start_scheduled_event")
    return await _transition(gateway, arguments, "start_scheduled_event", "start")


async def handle_end_scheduled_event(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "end_scheduled_event")
    return await _transition(gateway, arguments, "end_scheduled_event", "end")


async def handle_cancel_scheduled_event(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "cancel_scheduled_event")
    action = "cancel_scheduled_event"
    require_reason(arguments.get("reason"), action)
    return await _transition(gateway, arguments, action, "cancel")


async def handle_list_scheduled_event_users(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "list_scheduled_event_users")
    guild = await gateway.resolve_guild(arguments["server_id"])
    event = await _fetch_event(guild, arguments["event_id"], guild.name)
    limit = validate_limit(arguments.get("limit"), 100, 1000)
    users: List[Any] = []
    async for user in event.users(limit=limit):
        users.append(user_row(user))
    return json_text(
        {"eventId": str(event.id), "count": len(users), "users": users}
    )


async def handle_create_stage_instance(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_stage_instance")
    guild = await gateway.resolve_guild(arguments["server_id"])
    channel = _resolve_stage_channel(guild, arguments["channel_id"], guild.name)

    topic = str(arguments["topic"])
    privacy_level = _parse_privacy_level(arguments.get("privacy_level"))
    scheduled_event_id = None
    if arguments.get("scheduled_event_id") is not None:
        scheduled_event_id = validate_snowflake(arguments["scheduled_event_id"])
    send_start_notification = arguments.get("send_start_notification")
    reason = arguments.get("reason")

    action = "create_stage_instance"
    targets = {"server_id": str(arguments["server_id"]), "channel_id": str(channel.id)}
    if _is_dry_run(arguments):
        details = {
            "channelId": str(channel.id),
            "channelName": channel.name,
            "topic": topic,
        }
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs: Dict[str, Any] = {"topic": topic}
    if privacy_level is not None:
        kwargs["privacy_level"] = privacy_level
    if send_start_notification is not None:
        kwargs["send_start_notification"] = bool(send_start_notification)
    if scheduled_event_id is not None:
        kwargs["scheduled_event"] = discord.Object(id=scheduled_event_id)
    if reason is not None:
        kwargs["reason"] = reason
    instance = await channel.create_instance(**kwargs)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "instance": _stage_row(instance, channel),
        }
    )


async def handle_get_stage_instance(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_stage_instance")
    guild = await gateway.resolve_guild(arguments["server_id"])
    channel = _resolve_stage_channel(guild, arguments["channel_id"], guild.name)
    instance = await _fetch_instance(channel, guild.name)
    return json_text(_stage_row(instance, channel))


async def handle_edit_stage_instance(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_stage_instance")
    guild = await gateway.resolve_guild(arguments["server_id"])
    channel = _resolve_stage_channel(guild, arguments["channel_id"], guild.name)

    updates = _present(arguments, "topic")
    if arguments.get("privacy_level") is not None:
        updates["privacy_level"] = _parse_privacy_level(arguments["privacy_level"])
    if not updates:
        raise ValueError(
            "edit_stage_instance requires at least one of: topic, privacy_level"
        )
    reason = arguments.get("reason")

    instance = await _fetch_instance(channel, guild.name)
    action = "edit_stage_instance"
    targets = {"server_id": str(arguments["server_id"]), "channel_id": str(channel.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {"channelId": str(channel.id), "updates": sorted(updates)},
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    if reason is not None:
        updates["reason"] = reason
    await instance.edit(**updates)
    return json_text(
        {"status": "executed", "action": action, "channelId": str(channel.id)}
    )


async def handle_delete_stage_instance(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_stage_instance")
    guild = await gateway.resolve_guild(arguments["server_id"])
    channel = _resolve_stage_channel(guild, arguments["channel_id"], guild.name)
    action = "delete_stage_instance"
    reason = require_reason(arguments.get("reason"), action)
    instance = await _fetch_instance(channel, guild.name)

    targets = {"server_id": str(arguments["server_id"]), "channel_id": str(channel.id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {"channelId": str(channel.id), "topic": instance.topic},
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await instance.delete(reason=reason)
    return json_text(
        {"status": "executed", "action": action, "channelId": str(channel.id)}
    )
