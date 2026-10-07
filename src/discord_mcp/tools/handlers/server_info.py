import json
from typing import Any, Dict, List

import discord
from mcp.types import TextContent


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


async def handle_update_guild(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])

    updates: Dict[str, Any] = {}
    if "description" in arguments:
        description = arguments["description"]
        updates["description"] = None if description is None else str(description)
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
    if not updates:
        raise ValueError(
            "nothing to update: pass description, verification_level or explicit_content_filter"
        )

    reason = arguments.get("reason")
    await guild.edit(reason=reason, **updates)

    payload = {
        "status": "applied",
        "server_id": str(guild.id),
        "name": guild.name,
        "description": guild.description,
        "verification_level": str(guild.verification_level),
        "explicit_content_filter": str(guild.explicit_content_filter),
        "reason": reason,
    }
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
