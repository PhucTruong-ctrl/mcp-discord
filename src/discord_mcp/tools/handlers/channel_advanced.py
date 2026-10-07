"""Channel-advanced handlers: clone, create, follow, sync, voice status."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import discord
from mcp.types import TextContent

from discord_mcp.core.common import channel_row, json_text, require_gateway
from discord_mcp.core.permissions import (
    as_permission_bits,
    overwrite_index,
    overwrite_rows,
)
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token


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


def _resolve_category(guild: Any, category_id: Any, server: str):
    if not category_id:
        return None
    category = None
    if hasattr(guild, "get_channel"):
        try:
            category = guild.get_channel(int(category_id))
        except (TypeError, ValueError):
            category = None
    if category is None:
        for channel in getattr(guild, "channels", []):
            if str(getattr(channel, "id", "")) == str(category_id):
                category = channel
                break
    if category is None:
        raise ValueError(f"Category '{category_id}' not found in server '{server}'")
    if not _matches_channel_type(category, "category"):
        raise ValueError(
            f"Channel '{category_id}' in server '{server}' is a "
            f"{_channel_type(category)} channel, not a category"
        )
    return category


def _resolve_channel(guild: Any, channel_id: Any, server: str):
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


def _sorted_channels(guild: Any) -> List[Any]:
    return sorted(guild.channels, key=lambda c: (getattr(c, "position", 0), c.id))


def _present(arguments: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    """Kwargs the caller actually set — discord.py treats ``None`` as a real value."""
    return {key: arguments[key] for key in keys if arguments.get(key) is not None}


def _pair_ints(overwrite: Any) -> Optional[Tuple[int, ...]]:
    if overwrite is None:
        return None
    return tuple(as_permission_bits(bit) for bit in overwrite.pair())


def _sync_plan(category: Any, child: Any):
    """Read-side diff for one child: report rows plus the target ids to write.

    The applied state after a sync is exactly the category's overwrite set, so
    child targets absent from the category are reported as deletions.
    """
    cat_index = overwrite_index(category)
    child_index = overwrite_index(child)
    cat_pairs = {key: _pair_ints(value) for key, value in cat_index.items()}
    child_pairs = {key: _pair_ints(value) for key, value in child_index.items()}
    changed = sorted(
        key
        for key in set(cat_pairs) | set(child_pairs)
        if cat_pairs.get(key) != child_pairs.get(key)
    )
    entry = {
        "channelId": str(child.id),
        "channelName": getattr(child, "name", None),
        "previous": {row["targetId"]: row for row in overwrite_rows(child)},
        "applied": {row["targetId"]: row for row in overwrite_rows(category)},
        "changed": bool(changed),
    }
    return entry, cat_index, changed


def _overwrite_targets(*channels: Any) -> Dict[str, Any]:
    targets: Dict[str, Any] = {}
    for channel in channels:
        for target in (getattr(channel, "overwrites", None) or {}):
            targets[str(getattr(target, "id", target))] = target
    return targets


async def handle_clone_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "clone_channel")
    server_id = str(arguments["server_id"])
    guild = await gateway.resolve_guild(server_id)
    channel_id = str(arguments["channel_id"])
    channel = _resolve_channel(guild, channel_id, guild.name)
    if not callable(getattr(channel, "clone", None)):
        raise ValueError(
            f"Channel '{channel_id}' in server '{guild.name}' does not support cloning"
        )
    name = arguments.get("name") or channel.name
    reason = arguments.get("reason")

    action = "clone_channel"
    targets = {"channel_id": channel_id, "server_id": server_id}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"name": name, "reason": reason})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    cloned = await channel.clone(name=name, reason=reason)
    return json_text(
        {"status": "executed", "action": action, "channel": channel_row(cloned)}
    )


async def handle_create_announcement_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_announcement_channel")
    server_id = str(arguments["server_id"])
    guild = await gateway.resolve_guild(server_id)
    name = str(arguments["name"])
    category = _resolve_category(guild, arguments.get("category_id"), guild.name)
    reason = arguments.get("reason")

    action = "create_announcement_channel"
    targets = {"name": name, "server_id": server_id}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {"category_id": arguments.get("category_id"), "reason": reason},
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    channel = await guild.create_text_channel(
        name,
        news=True,
        category=category,
        reason=reason,
        **_present(
            arguments,
            "position",
            "topic",
            "slowmode_delay",
            "nsfw",
            "default_auto_archive_duration",
            "default_thread_slowmode_delay",
        ),
    )
    return json_text(
        {"status": "executed", "action": action, "channel": channel_row(channel)}
    )


def _video_quality_mode(value: Any) -> discord.VideoQualityMode:
    try:
        return discord.VideoQualityMode(int(value))
    except (TypeError, ValueError):
        raise ValueError(
            f"video_quality_mode must be 1 (auto) or 2 (full), got {value!r}"
        ) from None


async def handle_create_stage_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_stage_channel")
    server_id = str(arguments["server_id"])
    guild = await gateway.resolve_guild(server_id)
    name = str(arguments["name"])
    category = _resolve_category(guild, arguments.get("category_id"), guild.name)
    reason = arguments.get("reason")
    video_quality_mode = None
    if arguments.get("video_quality_mode") is not None:
        video_quality_mode = _video_quality_mode(arguments["video_quality_mode"])

    action = "create_stage_channel"
    targets = {"name": name, "server_id": server_id}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {"category_id": arguments.get("category_id"), "reason": reason},
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    options = _present(
        arguments, "position", "bitrate", "user_limit", "rtc_region", "nsfw"
    )
    if video_quality_mode is not None:
        options["video_quality_mode"] = video_quality_mode
    channel = await guild.create_stage_channel(
        name, category=category, reason=reason, **options
    )
    return json_text(
        {"status": "executed", "action": action, "channel": channel_row(channel)}
    )


async def handle_follow_channel(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "follow_channel")
    server_id = str(arguments["server_id"])
    guild = await gateway.resolve_guild(server_id)
    channel_id = str(arguments["channel_id"])
    webhook_channel_id = str(arguments["webhook_channel_id"])
    source = _resolve_channel(guild, channel_id, guild.name)
    destination = _resolve_channel(guild, webhook_channel_id, guild.name)

    is_news = getattr(source, "is_news", None)
    if not callable(is_news) or not is_news():
        raise ValueError(
            f"Channel '{channel_id}' in server '{guild.name}' must be an "
            "announcement (news) channel to be followed"
        )
    if not (
        _matches_channel_type(destination, "text")
        or _matches_channel_type(destination, "news")
    ):
        raise ValueError(
            f"Webhook channel '{webhook_channel_id}' in server '{guild.name}' "
            "must be a text channel"
        )

    action = "follow_channel"
    targets = {
        "channel_id": channel_id,
        "server_id": server_id,
        "webhook_channel_id": webhook_channel_id,
    }
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, {}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    webhook = await source.follow(destination=destination)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "webhookId": str(getattr(webhook, "id", None)),
            "webhookUrl": str(webhook.url),
        }
    )


async def handle_sync_channel_permissions(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "sync_channel_permissions")
    server_id = str(arguments["server_id"])
    guild = await gateway.resolve_guild(server_id)
    category_id = str(arguments["category_id"])
    category = _resolve_category(guild, category_id, guild.name)
    reason = arguments.get("reason")
    children = [
        channel
        for channel in _sorted_channels(guild)
        if getattr(channel, "id", None) != getattr(category, "id", None)
        and str(getattr(channel, "category_id", "")) == category_id
    ]

    action = "sync_channel_permissions"
    targets = {"category_id": category_id, "server_id": server_id}
    if _is_dry_run(arguments):
        diffs = [_sync_plan(category, child)[0] for child in children]
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {"categoryId": category_id, "diffs": diffs, "reason": reason},
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    updated: List[str] = []
    for child in children:
        _, cat_index, changed = _sync_plan(category, child)
        if not changed:
            continue
        targets_by_id = _overwrite_targets(category, child)
        for target_id in changed:
            overwrite = cat_index.get(target_id)
            if overwrite is None:
                await child.set_permissions(
                    targets_by_id[target_id], overwrite=None, reason=reason
                )
            else:
                await child.set_permissions(
                    targets_by_id[target_id],
                    overwrite=discord.PermissionOverwrite.from_pair(*overwrite.pair()),
                    reason=reason,
                )
        updated.append(str(child.id))

    return json_text(
        {
            "status": "executed",
            "action": action,
            "categoryId": category_id,
            "updated": sorted(updated),
            "unchanged": len(children) - len(updated),
        }
    )


async def handle_set_voice_channel_status(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "set_voice_channel_status")
    server_id = str(arguments["server_id"])
    guild = await gateway.resolve_guild(server_id)
    channel_id = str(arguments["channel_id"])
    channel = _resolve_channel(guild, channel_id, guild.name)

    status = arguments.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError(
            "status must be a non-empty string of 1-500 characters "
            "(whitespace-only is not allowed; Discord truncates at 500)"
        )
    if len(status) > 500:
        raise ValueError(f"status must be at most 500 characters, got {len(status)}")
    if not (
        _matches_channel_type(channel, "voice")
        or _matches_channel_type(channel, "stage")
    ):
        raise ValueError(
            f"Channel '{channel_id}' in server '{guild.name}' is a "
            f"{_channel_type(channel)} channel; voice channel status is only "
            "supported on voice and stage channels"
        )
    reason = arguments.get("reason")

    action = "set_voice_channel_status"
    targets = {"channel_id": channel_id, "server_id": server_id, "status": status}
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, {"reason": reason}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await channel.edit(status=status, reason=reason)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "channelId": channel_id,
            "channelStatus": status,
        }
    )
