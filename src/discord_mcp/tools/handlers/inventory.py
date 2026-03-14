import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from mcp.types import TextContent


def _overwrites_map(channel: Any) -> Dict[str, Dict[str, int]]:
    mapped: Dict[str, Dict[str, int]] = {}
    for target, overwrite in channel.overwrites.items():
        allow, deny = overwrite.pair()
        mapped[str(target.id)] = {
            "targetId": str(target.id),
            "targetName": getattr(target, "name", str(target.id)),
            "allow": int(allow.value),
            "deny": int(deny.value),
        }
    return mapped


async def handle_get_channels_structured(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].fetch_guild(arguments["server_id"])
    payload = {
        "serverId": str(guild.id),
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
                "topic": getattr(channel, "topic", None),
            }
            for channel in guild.channels
        ],
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def handle_get_channel_hierarchy(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].fetch_guild(arguments["server_id"])
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
    guild = await deps["gateway"].fetch_guild(arguments["server_id"])
    roles = sorted(guild.roles, key=lambda role: role.position, reverse=True)
    payload = {
        "serverId": str(guild.id),
        "roles": [
            {
                "id": str(role.id),
                "name": role.name,
                "position": role.position,
                "managed": bool(getattr(role, "managed", False)),
            }
            for role in roles
        ],
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
    guild = await deps["gateway"].fetch_guild(arguments["server_id"])
    payload = {
        "server": {"id": str(guild.id), "name": guild.name},
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
        "roles": [
            {
                "id": str(role.id),
                "name": role.name,
                "position": role.position,
            }
            for role in sorted(guild.roles, key=lambda r: r.position, reverse=True)
        ],
    }
    return [TextContent(type="text", text=json.dumps(payload))]


async def handle_get_channel_type_counts(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].fetch_guild(arguments["server_id"])
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
    guild = await deps["gateway"].fetch_guild(arguments["server_id"])
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
