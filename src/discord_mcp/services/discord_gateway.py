from __future__ import annotations

import datetime
import re
from typing import Any, Callable, Dict, List, Optional

import discord

from discord_mcp.core.resolve import normalize_name, try_int

_AUDIT_ACTION_ALIASES: Dict[str, discord.AuditLogAction] = {}
for _action in discord.AuditLogAction:
    _AUDIT_ACTION_ALIASES[_action.name] = _action
    _AUDIT_ACTION_ALIASES[_action.name.replace("_", "")] = _action


def resolve_audit_action(action_type: Any) -> discord.AuditLogAction:
    """Resolve an audit-log action filter from a name, an alias or a numeric value.

    Accepts the discord.py enum name (``role_update``), a spaced/cased spelling
    (``ROLE_UPDATE``, ``roleUpdate``, ``role update``), a dotted name
    (``AuditLogAction.role_update``) or the raw ``AuditLogAction`` value (``31``).
    """
    if isinstance(action_type, bool):
        raise ValueError(f"Invalid audit log action type: {action_type}")

    if isinstance(action_type, (int, float)):
        return discord.enums.try_enum(discord.AuditLogAction, int(action_type))

    token = str(action_type).strip()
    if token.lstrip("+-").isdigit():
        return discord.enums.try_enum(discord.AuditLogAction, int(token))

    for prefix in ("auditlogaction.", "audit_log_action."):
        if token.lower().startswith(prefix):
            token = token[len(prefix) :]
            break
    normalized = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")

    action = _AUDIT_ACTION_ALIASES.get(normalized) or _AUDIT_ACTION_ALIASES.get(
        normalized.replace("_", "")
    )
    if action is None:
        raise ValueError(
            f"Invalid audit log action type: {action_type}. Use an AuditLogAction name "
            f"(e.g. role_update, member_role_update, channel_overwrite_update) or its "
            f"numeric value (e.g. 31)."
        )
    return action


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

        if self._default_guild_id:
            default_id = try_int(self._default_guild_id)
            if default_id is None:
                raise ValueError(
                    f"Configured default server '{self._default_guild_id}' is invalid"
                )
            guild = client.get_guild(default_id)
            if guild is not None:
                return guild
            guild = await client.fetch_guild(default_id)
            if guild is not None:
                return guild
            raise ValueError(
                f"Configured default server '{self._default_guild_id}' is not accessible"
            )

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

    async def resolve_forum_post(
        self, post_id: str, server_id: Optional[str] = None
    ) -> discord.Thread:
        """Resolve a forum post (which is a Thread)."""
        return await self.resolve_thread(post_id, server_id)

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

        kwargs = {"limit": limit}
        if action_type not in (None, ""):
            kwargs["action"] = resolve_audit_action(action_type)

        entries = []
        async for entry in guild.audit_logs(**kwargs):
            entries.append(entry)
        return entries

    async def bulk_delete_messages(
        self, channel_id: str, message_ids: List[str], reason: Optional[str] = None
    ) -> int:
        """Delete messages by id from a channel. Returns the number requested."""
        channel = await self.fetch_channel(channel_id)
        targets = [discord.Object(id=int(message_id)) for message_id in message_ids]
        if not targets:
            return 0
        await channel.delete_messages(targets, reason=reason)
        return len(targets)

    async def timeout_member(
        self,
        server_id: str,
        member_id: str,
        duration_minutes: int,
        reason: Optional[str] = None,
    ) -> None:
        """Time a member out for N minutes; N <= 0 removes an existing timeout."""
        guild = await self.resolve_guild(server_id)
        member = await guild.fetch_member(int(member_id))
        until = (
            datetime.timedelta(minutes=int(duration_minutes))
            if int(duration_minutes) > 0
            else None
        )
        await member.timeout(until, reason=reason)

    async def kick_member(
        self, server_id: str, member_id: str, reason: Optional[str] = None
    ) -> None:
        guild = await self.resolve_guild(server_id)
        member = await guild.fetch_member(int(member_id))
        await member.kick(reason=reason)

    async def ban_member(
        self,
        server_id: str,
        member_id: str,
        delete_message_days: int = 0,
        reason: Optional[str] = None,
    ) -> None:
        guild = await self.resolve_guild(server_id)
        await guild.ban(
            discord.Object(id=int(member_id)),
            reason=reason,
            delete_message_seconds=int(delete_message_days) * 86400,
        )

    async def unban_member(
        self, server_id: str, member_id: str, reason: Optional[str] = None
    ) -> None:
        guild = await self.resolve_guild(server_id)
        await guild.unban(discord.Object(id=int(member_id)), reason=reason)

    async def bulk_ban_members(
        self,
        server_id: str,
        member_ids: List[str],
        reason: Optional[str] = None,
        delete_message_days: int = 0,
    ) -> int:
        """Ban every id in one request (falls back to individual bans)."""
        guild = await self.resolve_guild(server_id)
        users = [discord.Object(id=int(member_id)) for member_id in member_ids]
        if not users:
            return 0
        await guild.bulk_ban(
            users,
            reason=reason,
            delete_message_seconds=int(delete_message_days) * 86400,
        )
        return len(users)

    async def prune_inactive_members(
        self, server_id: str, days: int, reason: Optional[str] = None
    ) -> Optional[int]:
        guild = await self.resolve_guild(server_id)
        return await guild.prune_members(
            days=int(days), compute_prune_count=True, reason=reason
        )

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
