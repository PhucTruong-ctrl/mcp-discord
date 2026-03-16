from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import discord

from discord_mcp.core.resolve import normalize_name, try_int


class DiscordGateway:
    def __init__(
        self, client_getter: Callable[[], Any], default_guild_id: Optional[str] = None
    ):
        self._client_getter = client_getter
        self._default_guild_id = default_guild_id

    @property
    def client(self):
        client = self._client_getter()
        if not client:
            raise RuntimeError("Discord client not ready")
        return client

    async def resolve_guild(self, server_id: Optional[str] = None):
        client = self.client

        if server_id:
            guild_id = try_int(server_id)
            if guild_id is not None:
                guild = client.get_guild(guild_id)
                if guild is not None:
                    return guild
                guild = await client.fetch_guild(guild_id)
                if guild is not None:
                    return guild

            matches = [
                guild
                for guild in client.guilds
                if guild.name.lower() == str(server_id).lower()
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                detail = ", ".join(f"{g.name} ({g.id})" for g in matches)
                raise ValueError(
                    f"Multiple servers found for '{server_id}'. Use server ID. Matches: {detail}"
                )

            available = ", ".join(f"{g.name} ({g.id})" for g in client.guilds)
            raise ValueError(f"Server '{server_id}' not found. Available: {available}")

        if self._default_guild_id:
            default_id = try_int(self._default_guild_id)
            if default_id is not None:
                guild = client.get_guild(default_id)
                if guild is not None:
                    return guild
                guild = await client.fetch_guild(default_id)
                if guild is not None:
                    return guild

        if len(client.guilds) == 1:
            return client.guilds[0]

        available = ", ".join(f"{g.name} ({g.id})" for g in client.guilds)
        raise ValueError(
            f"Server ID is required because bot is in multiple servers. Available: {available}"
        )

    async def resolve_forum_channel(
        self, channel_identifier: str, server_id: Optional[str] = None
    ):
        guild = await self.resolve_guild(server_id)

        channel_id = try_int(channel_identifier)
        if channel_id is not None:
            channel = guild.get_channel(channel_id)
            if channel is None:
                channel = await guild.fetch_channel(channel_id)
            if isinstance(channel, discord.ForumChannel):
                return channel

        normalized = normalize_name(channel_identifier)
        matches = [
            ch
            for ch in guild.channels
            if isinstance(ch, discord.ForumChannel)
            and normalize_name(ch.name) == normalized
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            detail = ", ".join(f"{ch.name} ({ch.id})" for ch in matches)
            raise ValueError(
                f"Multiple forum channels found for '{channel_identifier}'. Use channel ID. Matches: {detail}"
            )

        available = ", ".join(
            f"#{ch.name}"
            for ch in guild.channels
            if isinstance(ch, discord.ForumChannel)
        )
        raise ValueError(
            f"Forum channel '{channel_identifier}' not found in '{guild.name}'. Available forums: {available}"
        )

    async def resolve_text_or_thread_channel(
        self, channel_identifier: str, server_id: Optional[str] = None
    ):
        guild = await self.resolve_guild(server_id)
        client = self.client

        channel_id = try_int(channel_identifier)
        if channel_id is not None:
            channel = client.get_channel(channel_id)
            if channel is None:
                channel = await client.fetch_channel(channel_id)
            if isinstance(channel, (discord.TextChannel, discord.Thread)):
                if channel.guild.id != guild.id:
                    raise ValueError(
                        f"Channel '{channel_identifier}' is not in server '{guild.name}'"
                    )
                return channel

        normalized = normalize_name(channel_identifier)
        matches = [
            ch for ch in guild.text_channels if normalize_name(ch.name) == normalized
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

    async def resolve_thread(self, thread_id: str, server_id: Optional[str] = None):
        parsed_id = try_int(thread_id)
        if parsed_id is None:
            raise ValueError("thread_id must be a valid integer Discord ID")

        client = self.client
        channel = client.get_channel(parsed_id)
        if channel is None:
            channel = await client.fetch_channel(parsed_id)

        if not isinstance(channel, discord.Thread):
            raise ValueError(f"Channel '{thread_id}' is not a thread")

        if server_id:
            guild = await self.resolve_guild(server_id)
            if channel.guild.id != guild.id:
                raise ValueError(
                    f"Thread '{thread_id}' is not in server '{guild.name}'"
                )
            return channel, guild

        return channel, channel.guild

    async def fetch_channel(self, channel_id: str):
        return await self.client.fetch_channel(int(channel_id))

    async def fetch_guild(self, guild_id: str):
        return await self.client.fetch_guild(int(guild_id))

    async def resolve_member(self, user_id: str, server_id: Optional[str] = None):
        """Resolve a member in a guild. Falls back to default guild if not specified."""
        guild = await self.resolve_guild(server_id)
        user_id_int = try_int(user_id)
        if user_id_int is None:
            raise ValueError(f"Invalid user ID: {user_id}")

        try:
            member = await guild.fetch_member(user_id_int)
            return member
        except discord.NotFound:
            raise ValueError(f"User '{user_id}' not found in server '{guild.name}'")

    async def resolve_role(self, role_id: str, server_id: Optional[str] = None):
        """Resolve a role in a guild. Falls back to default guild if not specified."""
        guild = await self.resolve_guild(server_id)
        role_id_int = try_int(role_id)
        if role_id_int is None:
            raise ValueError(f"Invalid role ID: {role_id}")

        role = guild.get_role(role_id_int)
        if role is None:
            raise ValueError(f"Role '{role_id}' not found in server '{guild.name}'")
        return role

    async def fetch_webhook(self, webhook_id: str, token: str):
        """Fetch a webhook by ID and token."""
        webhook_id_int = try_int(webhook_id)
        if webhook_id_int is None:
            raise ValueError(f"Invalid webhook ID: {webhook_id}")

        try:
            webhook = await self.client.fetch_webhook(webhook_id_int, token=token)
            return webhook
        except discord.NotFound:
            raise ValueError(f"Webhook '{webhook_id}' not found")

    async def fetch_audit_entries(
        self, server_id: str, limit: int = 50, action_type: Optional[str] = None
    ):
        """Fetch audit log entries for a guild."""
        guild = await self.resolve_guild(server_id)

        action = None
        if action_type:
            try:
                action = discord.AuditLogAction[action_type.upper()]
            except KeyError:
                raise ValueError(f"Invalid audit log action type: {action_type}")

        entries = []
        async for entry in guild.audit_logs(limit=limit, action=action):
            entries.append(entry)
        return entries

    async def collect_forum_threads(
        self,
        forum_channel: discord.ForumChannel,
        include_archived: bool,
        max_archived_threads_scan: int,
    ):
        threads: List[discord.Thread] = list(forum_channel.threads)
        if include_archived:
            scanned = 0
            async for archived in forum_channel.archived_threads(
                limit=max_archived_threads_scan
            ):
                threads.append(archived)
                scanned += 1
                if scanned >= max_archived_threads_scan:
                    break
        unique: Dict[int, discord.Thread] = {thread.id: thread for thread in threads}
        return list(unique.values())
