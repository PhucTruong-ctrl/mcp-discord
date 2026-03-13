import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import wraps
from urllib.parse import unquote, urlparse

import aiohttp
import discord
from discord.ext import commands
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

from discord_mcp.tools.handlers import dispatch_tool_call


def _configure_windows_stdout_encoding():
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


_configure_windows_stdout_encoding()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord-mcp-server")

# Discord bot setup
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

DEFAULT_GUILD_ID = os.getenv("DEFAULT_GUILD_ID") or os.getenv("DISCORD_GUILD_ID")
MAX_ARCHIVED_THREADS_SCAN = max(100, int(os.getenv("DISCORD_FORUM_MAX_FETCH", "1000")))

# Initialize Discord bot with necessary intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Initialize MCP server
app = Server("discord-server")

# Store Discord client reference
discord_client = None


@bot.event
async def on_ready():
    global discord_client
    discord_client = bot
    logger.info(f"Logged in as {bot.user.name}")


# Helper function to ensure Discord client is ready
def require_discord_client(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not discord_client:
            raise RuntimeError("Discord client not ready")
        return await func(*args, **kwargs)

    return wrapper


def _try_int(value: Any) -> Optional[int]:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _normalize_name(value: str) -> str:
    return value.strip().lower().removeprefix("#")


async def _resolve_guild(server_id: Optional[str] = None) -> discord.Guild:
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

    if DEFAULT_GUILD_ID:
        default_id = _try_int(DEFAULT_GUILD_ID)
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
) -> discord.ForumChannel:
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
) -> discord.TextChannel | discord.Thread:
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


async def _resolve_thread(
    thread_id: str, server_id: Optional[str] = None
) -> tuple[discord.Thread, discord.Guild]:
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


def _serialize_attachment(attachment: discord.Attachment) -> Dict[str, Any]:
    return {
        "id": str(attachment.id),
        "name": attachment.filename,
        "url": attachment.url,
        "proxyUrl": attachment.proxy_url,
        "size": attachment.size,
        "contentType": attachment.content_type,
        "width": attachment.width,
        "height": attachment.height,
    }


def _serialize_embed(embed: discord.Embed) -> Dict[str, Any]:
    return {
        "title": embed.title,
        "description": embed.description,
        "url": embed.url,
        "image": embed.image.url if embed.image else None,
        "thumbnail": embed.thumbnail.url if embed.thumbnail else None,
    }


def _serialize_message(message: discord.Message) -> Dict[str, Any]:
    return {
        "messageId": str(message.id),
        "author": str(message.author),
        "content": message.content,
        "timestamp": message.created_at.isoformat(),
        "attachments": [_serialize_attachment(att) for att in message.attachments],
        "embeds": [_serialize_embed(embed) for embed in message.embeds],
    }


def _serialize_forum_tag(tag: discord.ForumTag) -> Dict[str, Any]:
    emoji_name = None
    if tag.emoji:
        emoji_name = getattr(tag.emoji, "name", str(tag.emoji))
    return {
        "id": str(tag.id),
        "name": tag.name,
        "emoji": emoji_name,
        "moderated": tag.moderated,
    }


async def _collect_forum_threads(
    forum_channel: discord.ForumChannel, include_archived: bool
) -> List[discord.Thread]:
    threads: List[discord.Thread] = list(forum_channel.threads)

    if include_archived:
        scanned = 0
        async for archived in forum_channel.archived_threads(
            limit=MAX_ARCHIVED_THREADS_SCAN
        ):
            threads.append(archived)
            scanned += 1
            if scanned >= MAX_ARCHIVED_THREADS_SCAN:
                break

    unique: Dict[int, discord.Thread] = {thread.id: thread for thread in threads}
    return list(unique.values())


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available Discord tools."""
    return [
        # Server Information Tools
        Tool(
            name="get_server_info",
            description="Get information about a Discord server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server (guild) ID",
                    }
                },
                "required": ["server_id"],
            },
        ),
        Tool(
            name="get_channels",
            description="Get a list of all channels in a Discord server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server (guild) ID",
                    }
                },
                "required": ["server_id"],
            },
        ),
        Tool(
            name="list_members",
            description="Get a list of members in a server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server (guild) ID",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of members to fetch",
                        "minimum": 1,
                        "maximum": 1000,
                    },
                },
                "required": ["server_id"],
            },
        ),
        # Role Management Tools
        Tool(
            name="add_role",
            description="Add a role to a user",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "Discord server ID"},
                    "user_id": {"type": "string", "description": "User to add role to"},
                    "role_id": {"type": "string", "description": "Role ID to add"},
                },
                "required": ["server_id", "user_id", "role_id"],
            },
        ),
        Tool(
            name="remove_role",
            description="Remove a role from a user",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "Discord server ID"},
                    "user_id": {
                        "type": "string",
                        "description": "User to remove role from",
                    },
                    "role_id": {"type": "string", "description": "Role ID to remove"},
                },
                "required": ["server_id", "user_id", "role_id"],
            },
        ),
        # Channel Management Tools
        Tool(
            name="create_text_channel",
            description="Create a new text channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {"type": "string", "description": "Discord server ID"},
                    "name": {"type": "string", "description": "Channel name"},
                    "category_id": {
                        "type": "string",
                        "description": "Optional category ID to place channel in",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional channel topic",
                    },
                },
                "required": ["server_id", "name"],
            },
        ),
        Tool(
            name="delete_channel",
            description="Delete a channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "ID of channel to delete",
                    },
                    "reason": {"type": "string", "description": "Reason for deletion"},
                },
                "required": ["channel_id"],
            },
        ),
        # Message Reaction Tools
        Tool(
            name="add_reaction",
            description="Add a reaction to a message",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel containing the message",
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Message to react to",
                    },
                    "emoji": {
                        "type": "string",
                        "description": "Emoji to react with (Unicode or custom emoji ID)",
                    },
                },
                "required": ["channel_id", "message_id", "emoji"],
            },
        ),
        Tool(
            name="add_multiple_reactions",
            description="Add multiple reactions to a message",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel containing the message",
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Message to react to",
                    },
                    "emojis": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Emoji to react with (Unicode or custom emoji ID)",
                        },
                        "description": "List of emojis to add as reactions",
                    },
                },
                "required": ["channel_id", "message_id", "emojis"],
            },
        ),
        Tool(
            name="remove_reaction",
            description="Remove a reaction from a message",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel containing the message",
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Message to remove reaction from",
                    },
                    "emoji": {
                        "type": "string",
                        "description": "Emoji to remove (Unicode or custom emoji ID)",
                    },
                },
                "required": ["channel_id", "message_id", "emoji"],
            },
        ),
        Tool(
            name="send_message",
            description="Send a message to a specific channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID",
                    },
                    "content": {"type": "string", "description": "Message content"},
                },
                "required": ["channel_id", "content"],
            },
        ),
        Tool(
            name="read_messages",
            description="Read recent messages from a channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Number of messages to fetch (max 100)",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["channel_id"],
            },
        ),
        Tool(
            name="edit_message",
            description="Edit a message sent by the bot",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID",
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Message ID to edit",
                    },
                    "content": {
                        "type": "string",
                        "description": "New message content",
                    },
                },
                "required": ["channel_id", "message_id", "content"],
            },
        ),
        Tool(
            name="read_forum_threads",
            description="Read active forum threads and recent posts",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID (optional when default guild is configured)",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Forum channel name or ID",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Number of threads to fetch (max 50)",
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "before": {
                        "type": "string",
                        "description": "Optional message ID for history pagination",
                    },
                },
                "required": ["channel"],
            },
        ),
        Tool(
            name="list_threads",
            description="List forum threads without reading full message history",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID (optional when default guild is configured)",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Forum channel name or ID",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of threads to return",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "include_archived": {
                        "type": "boolean",
                        "description": "Include archived threads",
                    },
                },
                "required": ["channel"],
            },
        ),
        Tool(
            name="search_threads",
            description="Search forum threads by title",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID (optional when default guild is configured)",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Forum channel name or ID",
                    },
                    "query": {
                        "type": "string",
                        "description": "Thread title query",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of threads to return",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "include_archived": {
                        "type": "boolean",
                        "description": "Include archived threads in search",
                    },
                    "exact_match": {
                        "type": "boolean",
                        "description": "Require exact thread title match",
                    },
                },
                "required": ["channel", "query"],
            },
        ),
        Tool(
            name="add_thread_tags",
            description="Add tags to a forum thread",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID (optional when default guild is configured)",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Forum channel name or ID",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID",
                    },
                    "tag_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of tag names to apply",
                    },
                },
                "required": ["channel", "thread_id", "tag_names"],
            },
        ),
        Tool(
            name="unarchive_thread",
            description="Unarchive (reopen) a forum thread",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID (optional when default guild is configured)",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional reason",
                    },
                },
                "required": ["thread_id"],
            },
        ),
        Tool(
            name="download_attachment",
            description="Download a Discord attachment URL to local filesystem",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Discord attachment URL",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional filename override",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Output directory (default: current working directory)",
                    },
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="get_user_info",
            description="Get information about a Discord user",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "Discord user ID"}
                },
                "required": ["user_id"],
            },
        ),
        Tool(
            name="moderate_message",
            description="Delete a message and optionally timeout the user",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel ID containing the message",
                    },
                    "message_id": {
                        "type": "string",
                        "description": "ID of message to moderate",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for moderation",
                    },
                    "timeout_minutes": {
                        "type": "number",
                        "description": "Optional timeout duration in minutes",
                        "minimum": 0,
                        "maximum": 40320,  # Max 4 weeks
                    },
                },
                "required": ["channel_id", "message_id", "reason"],
            },
        ),
        Tool(
            name="list_servers",
            description="Get a list of all Discord servers the bot has access to with their details such as name, id, member count, and creation date.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


@app.call_tool()
@require_discord_client
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle Discord tool calls."""

    arguments = arguments or {}
    deps = {
        "discord_client": discord_client,
        "try_int": _try_int,
        "normalize_name": _normalize_name,
        "resolve_forum_channel": _resolve_forum_channel,
        "resolve_text_or_thread_channel": _resolve_text_or_thread_channel,
        "resolve_thread": _resolve_thread,
        "serialize_message": _serialize_message,
        "serialize_forum_tag": _serialize_forum_tag,
        "collect_forum_threads": _collect_forum_threads,
    }
    return await dispatch_tool_call(name, arguments, deps)


async def main():
    # Start Discord bot in the background
    asyncio.create_task(bot.start(DISCORD_TOKEN))

    # Run MCP server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
