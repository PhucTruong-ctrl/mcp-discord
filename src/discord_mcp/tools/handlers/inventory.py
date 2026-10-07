import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import discord
from mcp.types import TextContent

from discord_mcp.core.emoji import emoji_payload
from discord_mcp.core.permissions import overwrite_rows, role_payload
from discord_mcp.core.serialize import _serialize_forum_tag


def _overwrites_map(channel: Any) -> Dict[str, Dict[str, Any]]:
    return {row["targetId"]: row for row in overwrite_rows(channel)}


def _structured_channel(channel: Any) -> Dict[str, Any]:
    """Channel row; forum channels also carry the fields update_forum_channel accepts."""
    row: Dict[str, Any] = {
        "id": str(channel.id),
        "name": channel.name,
        "type": str(channel.type),
        "position": getattr(channel, "position", 0),
        "categoryId": (
            str(channel.category_id)
            if getattr(channel, "category_id", None) is not None
            else None
        ),
        "topic": getattr(channel, "topic", None),
    }
    channel_type = str(getattr(channel, "type", ""))
    is_forum = "forum" in channel_type
    tags = getattr(channel, "available_tags", None) if is_forum else None
    if is_forum and tags is not None:
        row["availableTags"] = [_serialize_forum_tag(tag) for tag in tags]
        row["defaultReactionEmoji"] = emoji_payload(
            getattr(channel, "default_reaction_emoji", None)
        )
        sort_order = getattr(channel, "default_sort_order", None)
        row["defaultSortOrder"] = (
            sort_order.value if hasattr(sort_order, "value") else sort_order
        )
        row["defaultAutoArchiveDuration"] = getattr(
            channel, "default_auto_archive_duration", None
        )
        row["nsfw"] = bool(getattr(channel, "nsfw", False))
        row["slowmodeDelay"] = getattr(channel, "slowmode_delay", None)
        layout = getattr(channel, "default_layout", None)
        row["defaultLayout"] = layout.value if hasattr(layout, "value") else layout
    return row


async def handle_get_channels_structured(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(arguments["server_id"])
    payload = {
        "serverId": str(guild.id),
        "channels": [_structured_channel(channel) for channel in guild.channels],
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def handle_get_channel_hierarchy(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(arguments["server_id"])
    categories = []
    top_level = []

    sorted_channels = sorted(guild.channels, key=lambda ch: getattr(ch, "position", 0))
    for channel in sorted_channels:
        category_id = getattr(channel, "category_id", None)
        item = {
            "id": str(channel.id),
            "name": channel.name,
            "type": str(channel.type),
            "position": getattr(channel, "position", 0),
        }
        if str(getattr(channel, "type", "")) == "category":
            children = [
                {
                    "id": str(child.id),
                    "name": child.name,
                    "type": str(child.type),
                    "position": getattr(child, "position", 0),
                }
                for child in sorted_channels
                if getattr(child, "category_id", None) == channel.id
            ]
            item["children"] = children
            categories.append(item)
        elif category_id is None:
            top_level.append(item)

    payload = {
        "serverId": str(guild.id),
        "categories": categories,
        "topLevel": top_level,
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def handle_get_role_hierarchy(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(arguments["server_id"])
    roles = sorted(guild.roles, key=lambda role: role.position, reverse=True)
    payload = {
        "serverId": str(guild.id),
        "everyoneRoleId": str(
            getattr(getattr(guild, "default_role", None), "id", guild.id)
        ),
        "roleCount": len(roles),
        "roles": [role_payload(role) for role in roles],
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def handle_get_permission_overwrites(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    channel = await deps["gateway"].fetch_channel(arguments["channel_id"])
    payload = {
        "channelId": str(channel.id),
        "overwrites": list(_overwrites_map(channel).values()),
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def handle_diff_channel_permissions(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    source = await deps["gateway"].fetch_channel(arguments["source_channel_id"])
    target = await deps["gateway"].fetch_channel(arguments["target_channel_id"])

    source_map = _overwrites_map(source)
    target_map = _overwrites_map(target)
    all_targets = sorted(set(source_map.keys()) | set(target_map.keys()))

    diffs = []
    for target_id in all_targets:
        source_overwrite = source_map.get(target_id)
        target_overwrite = target_map.get(target_id)
        if source_overwrite != target_overwrite:
            diffs.append(
                {
                    "targetId": target_id,
                    "source": source_overwrite,
                    "target": target_overwrite,
                }
            )

    payload = {
        "sourceChannelId": str(source.id),
        "targetChannelId": str(target.id),
        "diffCount": len(diffs),
        "diffs": diffs,
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def handle_export_server_snapshot(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(arguments["server_id"])
    roles = sorted(guild.roles, key=lambda role: role.position, reverse=True)
    payload = {
        "snapshotVersion": 2,
        "server": {"id": str(guild.id), "name": guild.name},
        "everyoneRoleId": str(
            getattr(getattr(guild, "default_role", None), "id", guild.id)
        ),
        "channels": [
            {
                "id": str(channel.id),
                "name": channel.name,
                "type": str(channel.type),
                "position": getattr(channel, "position", 0),
                "categoryId": (
                    str(channel.category_id)
                    if getattr(channel, "category_id", None) is not None
                    else None
                ),
            }
            for channel in guild.channels
        ],
        "roleCount": len(roles),
        # role rows carry the permission bitfield (+ decoded names) so this payload can
        # be fed straight back into permission_drift_check as baseline_snapshot
        "roles": [role_payload(role) for role in roles],
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def handle_get_channel_type_counts(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(arguments["server_id"])
    counts = Counter(str(channel.type) for channel in guild.channels)
    payload = {
        "serverId": str(guild.id),
        "counts": dict(counts),
        "totalChannels": sum(counts.values()),
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def handle_list_inactive_channels(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(arguments["server_id"])
    days = int(arguments.get("days", 30))
    threshold = datetime.now(timezone.utc) - timedelta(days=days)

    inactive = []
    for channel in guild.text_channels:
        last_message_at = None
        async for message in channel.history(limit=1):
            last_message_at = message.created_at
            break

        if last_message_at is None or last_message_at < threshold:
            inactive.append(
                {
                    "id": str(channel.id),
                    "name": channel.name,
                    "lastMessageAt": (
                        last_message_at.isoformat()
                        if last_message_at is not None
                        else None
                    ),
                }
            )

    payload = {
        "serverId": str(guild.id),
        "days": days,
        "inactive": inactive,
    }
    return [TextContent(type="text", text=json.dumps(payload))]


def _permission_mask(values: Any, label: str) -> int:
    """Coerce a list of permission names (or raw bit values) into a bitfield."""
    if values is None:
        return 0
    if isinstance(values, (str, int, bool)) or not isinstance(
        values, (list, tuple, set)
    ):
        raise ValueError(f"{label} must be an array of permission names or bit values")

    mask = 0
    unknown = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError(f"{label} entries must be permission names or bit values")
        if isinstance(item, int) or str(item).strip().isdigit():
            mask |= int(item)
            continue
        key = str(item).strip().lower().replace("-", "_").replace(" ", "_")
        bit = discord.Permissions.VALID_FLAGS.get(key)
        if bit is None:
            unknown.append(str(item))
            continue
        mask |= bit

    if unknown:
        raise ValueError(f"unknown {label} permission(s): {', '.join(sorted(unknown))}")
    return mask


async def _role_target(guild: Any, role_id: int) -> Any:
    if role_id == guild.id:
        return guild.default_role
    role = guild.get_role(role_id)
    if role is None:
        role = next((r for r in await guild.fetch_roles() if r.id == role_id), None)
    if role is None:
        raise ValueError(f"role '{role_id}' not found in '{guild.name}'")
    return role


async def _member_target(guild: Any, member_id: int) -> Any:
    member = guild.get_member(member_id)
    if member is None:
        member = await guild.fetch_member(member_id)
    return member


async def _resolve_overwrite_target(
    guild: Any, target_id: Any, target_type: Any = None
) -> Any:
    """Resolve a role/member object for a channel permission overwrite."""
    entity_id = int(target_id)
    kind = str(target_type).strip().lower() if target_type else None

    if kind in ("role", "r"):
        return await _role_target(guild, entity_id)
    if kind in ("member", "user", "m"):
        return await _member_target(guild, entity_id)
    if kind is not None:
        raise ValueError("target_type must be 'role' or 'member'")

    if entity_id == guild.id:
        return guild.default_role
    role = guild.get_role(entity_id)
    if role is not None:
        return role
    member = guild.get_member(entity_id)
    if member is not None:
        return member
    raise ValueError(
        f"target '{target_id}' is not a cached role or member; pass target_type to fetch it"
    )


async def _overwrite_channel(arguments: Dict[str, Any], deps: Dict[str, Any]):
    channel = await deps["gateway"].fetch_channel(arguments["channel_id"])
    guild = getattr(channel, "guild", None)
    if guild is None:
        raise ValueError(
            f"could not resolve the guild for channel '{arguments['channel_id']}'"
        )
    return channel, guild


async def handle_set_channel_permission_overwrite(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    channel, guild = await _overwrite_channel(arguments, deps)
    target = await _resolve_overwrite_target(
        guild, arguments["target_id"], arguments.get("target_type")
    )

    allow = _permission_mask(arguments.get("allow"), "allow")
    deny = _permission_mask(arguments.get("deny"), "deny")
    if allow == 0 and deny == 0:
        raise ValueError(
            "allow and deny are both empty; use remove_channel_permission_overwrite "
            "to delete an existing overwrite"
        )

    overwrite = discord.PermissionOverwrite.from_pair(
        discord.Permissions(allow), discord.Permissions(deny)
    )
    reason = arguments.get("reason")
    await channel.set_permissions(target, overwrite=overwrite, reason=reason)

    refreshed = await deps["gateway"].fetch_channel(arguments["channel_id"])
    payload = {
        "status": "applied",
        "channelId": str(channel.id),
        "channelName": getattr(channel, "name", None),
        "targetId": str(target.id),
        "targetName": getattr(target, "name", None),
        "overwrites": list(_overwrites_map(refreshed).values()),
        "reason": reason,
    }
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def handle_remove_channel_permission_overwrite(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    channel, guild = await _overwrite_channel(arguments, deps)
    target = await _resolve_overwrite_target(
        guild, arguments["target_id"], arguments.get("target_type")
    )

    reason = arguments.get("reason")
    await channel.set_permissions(target, overwrite=None, reason=reason)

    refreshed = await deps["gateway"].fetch_channel(arguments["channel_id"])
    payload = {
        "status": "applied",
        "channelId": str(channel.id),
        "targetId": str(target.id),
        "overwrites": list(_overwrites_map(refreshed).values()),
        "reason": reason,
    }
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
