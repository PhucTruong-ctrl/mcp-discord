from typing import Any, Dict, List, Optional

import discord

from discord_mcp.core.runtime import (
    get_default_guild_id,
    get_discord_client,
    get_max_archived_threads_scan,
)


def _try_int(value: Any) -> Optional[int]:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _normalize_name(value: str) -> str:
    return value.strip().lower().removeprefix("#")


async def _resolve_guild(server_id: Optional[str] = None):
    discord_client = get_discord_client()
    if server_id:
        guild_id = _try_int(server_id)
        if guild_id is not None:
            guild = discord_client.get_guild(guild_id)
            if guild is not None:
                return guild
            guild = await discord_client.fetch_guild(guild_id)
            if guild is not None:
                return guild

        matches = [
            guild
            for guild in discord_client.guilds
            if guild.name.lower() == str(server_id).lower()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            detail = ", ".join(f"{g.name} ({g.id})" for g in matches)
            raise ValueError(
                f"Multiple servers found for '{server_id}'. Use server ID. Matches: {detail}"
            )

        available = ", ".join(f"{g.name} ({g.id})" for g in discord_client.guilds)
        raise ValueError(f"Server '{server_id}' not found. Available: {available}")

    default_guild_id = get_default_guild_id()
    if default_guild_id:
        default_id = _try_int(default_guild_id)
        if default_id is not None:
            guild = discord_client.get_guild(default_id)
            if guild is not None:
                return guild
            guild = await discord_client.fetch_guild(default_id)
            if guild is not None:
                return guild

    if len(discord_client.guilds) == 1:
        return discord_client.guilds[0]

    available = ", ".join(f"{g.name} ({g.id})" for g in discord_client.guilds)
    raise ValueError(
        f"Server ID is required because bot is in multiple servers. Available: {available}"
    )


async def _resolve_forum_channel(
    channel_identifier: str, server_id: Optional[str] = None
):
    guild = await _resolve_guild(server_id)

    channel_id = _try_int(channel_identifier)
    if channel_id is not None:
        channel = guild.get_channel(channel_id)
        if channel is None:
            channel = await guild.fetch_channel(channel_id)
        if isinstance(channel, discord.ForumChannel):
            return channel

    normalized = _normalize_name(channel_identifier)
    matches = [
        ch
        for ch in guild.channels
        if isinstance(ch, discord.ForumChannel)
        and _normalize_name(ch.name) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        detail = ", ".join(f"{ch.name} ({ch.id})" for ch in matches)
        raise ValueError(
            f"Multiple forum channels found for '{channel_identifier}'. Use channel ID. Matches: {detail}"
        )

    available = ", ".join(
        f"#{ch.name}" for ch in guild.channels if isinstance(ch, discord.ForumChannel)
    )
    raise ValueError(
        f"Forum channel '{channel_identifier}' not found in '{guild.name}'. Available forums: {available}"
    )


async def _resolve_text_or_thread_channel(
    channel_identifier: str, server_id: Optional[str] = None
):
    discord_client = get_discord_client()
    guild = await _resolve_guild(server_id)

    channel_id = _try_int(channel_identifier)
    if channel_id is not None:
        channel = discord_client.get_channel(channel_id)
        if channel is None:
            channel = await discord_client.fetch_channel(channel_id)

        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            if channel.guild.id != guild.id:
                raise ValueError(
                    f"Channel '{channel_identifier}' is not in server '{guild.name}'"
                )
            return channel

    normalized = _normalize_name(channel_identifier)
    matches = [
        ch for ch in guild.text_channels if _normalize_name(ch.name) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        detail = ", ".join(f"#{ch.name} ({ch.id})" for ch in matches)
        raise ValueError(
            f"Multiple channels found for '{channel_identifier}'. Use channel ID. Matches: {detail}"
        )

    available = ", ".join(f"#{ch.name}" for ch in guild.text_channels)
    raise ValueError(
        f"Text channel '{channel_identifier}' not found in '{guild.name}'. Available channels: {available}"
    )


async def _resolve_thread(thread_id: str, server_id: Optional[str] = None):
    discord_client = get_discord_client()
    parsed_id = _try_int(thread_id)
    if parsed_id is None:
        raise ValueError("thread_id must be a valid integer Discord ID")

    channel = discord_client.get_channel(parsed_id)
    if channel is None:
        channel = await discord_client.fetch_channel(parsed_id)

    if not isinstance(channel, discord.Thread):
        raise ValueError(f"Channel '{thread_id}' is not a thread")

    if server_id:
        guild = await _resolve_guild(server_id)
        if channel.guild.id != guild.id:
            raise ValueError(f"Thread '{thread_id}' is not in server '{guild.name}'")
        return channel, guild

    return channel, channel.guild


async def _collect_forum_threads(
    forum_channel: Any, include_archived: bool
) -> List[Any]:
    threads: List[Any] = list(forum_channel.threads)

    if include_archived:
        scanned = 0
        max_archived_threads_scan = get_max_archived_threads_scan()
        async for archived in forum_channel.archived_threads(
            limit=max_archived_threads_scan
        ):
            threads.append(archived)
            scanned += 1
            if scanned >= max_archived_threads_scan:
                break

    unique: Dict[int, Any] = {thread.id: thread for thread in threads}
    return list(unique.values())
