import json
from typing import Any, Dict, List

from mcp.types import TextContent


def _sorted_channels(guild):
    return sorted(guild.channels, key=lambda c: (getattr(c, "position", 0), c.id))


async def handle_topology_channel_tree(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])

    categories = {}
    roots = []
    for channel in _sorted_channels(guild):
        if getattr(channel, "type", None) == "category":
            categories[channel.id] = {
                "id": str(channel.id),
                "name": channel.name,
                "children": [],
            }
        else:
            parent_id = getattr(channel, "category_id", None)
            entry = {
                "id": str(channel.id),
                "name": channel.name,
                "type": str(getattr(channel, "type", "unknown")),
            }
            if parent_id and parent_id in categories:
                categories[parent_id]["children"].append(entry)
            else:
                roots.append(entry)

    payload = {
        "server": {"id": str(guild.id), "name": guild.name},
        "categories": list(categories.values()),
        "rootChannels": roots,
    }
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def handle_topology_channel_children(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    category_id = int(arguments["category_id"])

    children = []
    for channel in _sorted_channels(guild):
        if getattr(channel, "category_id", None) == category_id:
            children.append(
                {
                    "id": str(channel.id),
                    "name": channel.name,
                    "type": str(getattr(channel, "type", "unknown")),
                }
            )

    payload = {
        "server": {"id": str(guild.id), "name": guild.name},
        "categoryId": str(category_id),
        "children": children,
    }
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def handle_topology_role_hierarchy(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    roles = sorted(guild.roles, key=lambda r: r.position, reverse=True)

    payload = {
        "server": {"id": str(guild.id), "name": guild.name},
        "roles": [
            {
                "id": str(role.id),
                "name": role.name,
                "position": role.position,
                "hoist": bool(getattr(role, "hoist", False)),
                "mentionable": bool(getattr(role, "mentionable", False)),
            }
            for role in roles
        ],
    }
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def handle_topology_permission_matrix(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    filter_ids = {str(v) for v in arguments.get("channel_ids", [])}

    channels = []
    for channel in _sorted_channels(guild):
        if filter_ids and str(channel.id) not in filter_ids:
            continue
        channels.append(
            {
                "id": str(channel.id),
                "name": channel.name,
                "type": str(getattr(channel, "type", "unknown")),
                "overwrites": len(getattr(channel, "overwrites", {}) or {}),
            }
        )

    payload = {
        "server": {"id": str(guild.id), "name": guild.name},
        "roleCount": len(guild.roles),
        "channelCount": len(channels),
        "channels": channels,
    }
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
