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

    if name in {"send_message", "send-message"}:
        server_id = arguments.get("server_id") or arguments.get("server")
        channel_identifier = arguments.get("channel_id") or arguments.get("channel")
        content = arguments.get("content") or arguments.get("message")

        if not channel_identifier:
            raise ValueError("channel_id (or channel) is required")
        if content is None:
            raise ValueError("content (or message) is required")

        channel = await _resolve_text_or_thread_channel(
            str(channel_identifier), server_id
        )
        message = await channel.send(str(content))
        return [
            TextContent(
                type="text", text=f"Message sent successfully. Message ID: {message.id}"
            )
        ]

    elif name in {"read_messages", "read-messages"}:
        server_id = arguments.get("server_id") or arguments.get("server")
        channel_identifier = arguments.get("channel_id") or arguments.get("channel")
        if not channel_identifier:
            raise ValueError("channel_id (or channel) is required")

        channel = await _resolve_text_or_thread_channel(
            str(channel_identifier), server_id
        )
        limit = min(int(arguments.get("limit", 10)), 100)
        before = arguments.get("before")
        before_obj = discord.Object(id=int(before)) if _try_int(before) else None
        messages = []
        async for message in channel.history(limit=limit, before=before_obj):
            reaction_data = []
            for reaction in message.reactions:
                emoji_str = (
                    str(reaction.emoji.name)
                    if hasattr(reaction.emoji, "name") and reaction.emoji.name
                    else str(reaction.emoji.id)
                    if hasattr(reaction.emoji, "id")
                    else str(reaction.emoji)
                )
                reaction_info = {"emoji": emoji_str, "count": reaction.count}
                reaction_data.append(reaction_info)
            messages.append(
                {
                    "id": str(message.id),
                    "author": str(message.author),
                    "content": message.content,
                    "timestamp": message.created_at.isoformat(),
                    "reactions": reaction_data,  # Add reactions to message dict
                }
            )

        # Helper function to format reactions
        def format_reaction(r):
            return f"{r['emoji']}({r['count']})"

        return [
            TextContent(
                type="text",
                text=f"Retrieved {len(messages)} messages:\n\n"
                + "\n".join(
                    [
                        f"{m['author']} ({m['timestamp']}): {m['content']}\n"
                        + f"Reactions: {', '.join([format_reaction(r) for r in m['reactions']]) if m['reactions'] else 'No reactions'}"
                        for m in messages
                    ]
                ),
            )
        ]

    elif name in {"edit_message", "edit-message"}:
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.fetch_message(int(arguments["message_id"]))
        await message.edit(content=arguments["content"])
        return [
            TextContent(
                type="text",
                text=f"Message edited successfully. Message ID: {message.id}",
            )
        ]

    elif name in {"read_forum_threads", "read-forum-threads"}:
        server_id = arguments.get("server_id") or arguments.get("server")
        channel_identifier = arguments["channel"]
        limit = min(int(arguments.get("limit", 10)), 50)
        before = arguments.get("before")
        before_obj = discord.Object(id=int(before)) if _try_int(before) else None

        forum_channel = await _resolve_forum_channel(channel_identifier, server_id)
        threads = sorted(
            list(forum_channel.threads),
            key=lambda t: t.created_at or discord.utils.utcnow(),
            reverse=True,
        )[:limit]

        thread_payload = []
        for thread in threads:
            messages = []
            async for msg in thread.history(limit=10, before=before_obj):
                payload = _serialize_message(msg)
                payload.update(
                    {
                        "thread": thread.name,
                        "threadId": str(thread.id),
                        "channel": f"#{forum_channel.name}",
                        "server": thread.guild.name,
                    }
                )
                messages.append(payload)

            thread_payload.append(
                {
                    "thread": thread.name,
                    "threadId": str(thread.id),
                    "tags": [_serialize_forum_tag(tag) for tag in thread.applied_tags],
                    "messages": messages,
                }
            )

        return [
            TextContent(
                type="text",
                text=json.dumps(thread_payload, ensure_ascii=False, indent=2),
            )
        ]

    elif name in {"list_threads", "list-threads"}:
        server_id = arguments.get("server_id") or arguments.get("server")
        channel_identifier = arguments["channel"]
        limit = min(int(arguments.get("limit", 50)), 100)
        include_archived = bool(
            arguments.get("include_archived", arguments.get("includeArchived", False))
        )

        forum_channel = await _resolve_forum_channel(channel_identifier, server_id)
        all_threads = await _collect_forum_threads(forum_channel, include_archived)
        thread_list = sorted(
            all_threads,
            key=lambda t: t.created_at or discord.utils.utcnow(),
            reverse=True,
        )[:limit]

        result = {
            "forumChannel": forum_channel.name,
            "server": forum_channel.guild.name,
            "includeArchived": include_archived,
            "totalThreads": len(thread_list),
            "threads": [
                {
                    "threadId": str(thread.id),
                    "threadName": thread.name,
                    "createdAt": thread.created_at.isoformat()
                    if thread.created_at
                    else None,
                    "ownerId": str(thread.owner_id) if thread.owner_id else None,
                    "archived": thread.archived,
                    "locked": thread.locked,
                    "messageCount": thread.message_count,
                    "memberCount": thread.member_count,
                    "totalMessageSent": thread.total_message_sent,
                    "rateLimitPerUser": thread.slowmode_delay,
                    "tags": [_serialize_forum_tag(tag) for tag in thread.applied_tags],
                    "lastMessageId": str(thread.last_message_id)
                    if thread.last_message_id
                    else None,
                }
                for thread in thread_list
            ],
        }

        return [
            TextContent(
                type="text", text=json.dumps(result, ensure_ascii=False, indent=2)
            )
        ]

    elif name in {"search_threads", "search-threads"}:
        server_id = arguments.get("server_id") or arguments.get("server")
        channel_identifier = arguments["channel"]
        query = str(arguments["query"]).strip()
        limit = min(int(arguments.get("limit", 50)), 100)
        include_archived = bool(
            arguments.get("include_archived", arguments.get("includeArchived", True))
        )
        exact_match = bool(
            arguments.get("exact_match", arguments.get("exactMatch", False))
        )

        forum_channel = await _resolve_forum_channel(channel_identifier, server_id)
        all_threads = await _collect_forum_threads(forum_channel, include_archived)
        query_lower = query.lower()

        def matches(thread: discord.Thread) -> bool:
            name_lower = thread.name.lower()
            if exact_match:
                return name_lower == query_lower
            return query_lower in name_lower

        filtered = [thread for thread in all_threads if matches(thread)]
        thread_list = sorted(
            filtered,
            key=lambda t: t.created_at or discord.utils.utcnow(),
            reverse=True,
        )[:limit]

        result = {
            "forumChannel": forum_channel.name,
            "server": forum_channel.guild.name,
            "query": query,
            "exactMatch": exact_match,
            "totalFound": len(thread_list),
            "totalMatched": len(filtered),
            "includeArchived": include_archived,
            "threads": [
                {
                    "threadId": str(thread.id),
                    "threadName": thread.name,
                    "createdAt": thread.created_at.isoformat()
                    if thread.created_at
                    else None,
                    "ownerId": str(thread.owner_id) if thread.owner_id else None,
                    "archived": thread.archived,
                    "locked": thread.locked,
                    "messageCount": thread.message_count,
                    "memberCount": thread.member_count,
                    "totalMessageSent": thread.total_message_sent,
                    "rateLimitPerUser": thread.slowmode_delay,
                    "tags": [_serialize_forum_tag(tag) for tag in thread.applied_tags],
                    "lastMessageId": str(thread.last_message_id)
                    if thread.last_message_id
                    else None,
                }
                for thread in thread_list
            ],
        }

        return [
            TextContent(
                type="text", text=json.dumps(result, ensure_ascii=False, indent=2)
            )
        ]

    elif name in {"add_thread_tags", "add-thread-tags"}:
        server_id = arguments.get("server_id") or arguments.get("server")
        channel_identifier = arguments["channel"]
        thread_id = arguments.get("thread_id") or arguments.get("threadId")
        tag_names = arguments.get("tag_names") or arguments.get("tagNames")

        if not thread_id:
            raise ValueError("thread_id is required")
        if not isinstance(tag_names, list) or not tag_names:
            raise ValueError("tag_names must be a non-empty array")

        forum_channel = await _resolve_forum_channel(channel_identifier, server_id)
        thread, _ = await _resolve_thread(str(thread_id), server_id)

        if thread.parent_id != forum_channel.id:
            raise ValueError(
                f"Thread '{thread.id}' does not belong to forum channel '{forum_channel.name}'"
            )

        requested = {_normalize_name(tag): tag for tag in tag_names}
        tag_lookup = {
            _normalize_name(tag.name): tag for tag in forum_channel.available_tags
        }

        missing = [
            original for key, original in requested.items() if key not in tag_lookup
        ]
        if missing:
            available = ", ".join(tag.name for tag in forum_channel.available_tags)
            raise ValueError(
                f"Tags not found: {', '.join(missing)}. Available tags: {available}"
            )

        merged: Dict[int, discord.ForumTag] = {
            tag.id: tag for tag in thread.applied_tags
        }
        for key in requested:
            tag = tag_lookup[key]
            merged[tag.id] = tag

        await thread.edit(applied_tags=list(merged.values()))
        applied_names = [tag.name for tag in merged.values()]
        return [
            TextContent(
                type="text",
                text=f"Tags added to thread '{thread.name}'. Applied tags: {', '.join(applied_names)}",
            )
        ]

    elif name in {"unarchive_thread", "unarchive-thread"}:
        server_id = arguments.get("server_id") or arguments.get("server")
        thread_id = arguments.get("thread_id") or arguments.get("threadId")
        reason = arguments.get("reason")

        if not thread_id:
            raise ValueError("thread_id is required")

        thread, _ = await _resolve_thread(str(thread_id), server_id)
        if not thread.archived:
            return [
                TextContent(
                    type="text",
                    text=f"Thread '{thread.name}' is already active.",
                )
            ]

        await thread.edit(archived=False, reason=reason)
        return [
            TextContent(
                type="text",
                text=f"Thread '{thread.name}' has been unarchived.",
            )
        ]

    elif name in {"download_attachment", "download-attachment"}:
        url = arguments["url"]
        filename = arguments.get("filename")
        directory = arguments.get("directory")

        if not filename:
            parsed = urlparse(url)
            filename = unquote(Path(parsed.path).name) or "downloaded_file"

        target_dir = Path(directory) if directory else Path.cwd()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status >= 400:
                    raise ValueError(
                        f"Failed to download attachment: HTTP {response.status}"
                    )
                data = await response.read()

        target_path.write_bytes(data)
        return [
            TextContent(
                type="text",
                text=(
                    "Attachment downloaded successfully. "
                    f"Path: {target_path} ({len(data)} bytes)"
                ),
            )
        ]

    elif name == "get_user_info":
        user = await discord_client.fetch_user(int(arguments["user_id"]))
        user_info = {
            "id": str(user.id),
            "name": user.name,
            "discriminator": user.discriminator,
            "bot": user.bot,
            "created_at": user.created_at.isoformat(),
        }
        return [
            TextContent(
                type="text",
                text=f"User information:\n"
                + f"Name: {user_info['name']}#{user_info['discriminator']}\n"
                + f"ID: {user_info['id']}\n"
                + f"Bot: {user_info['bot']}\n"
                + f"Created: {user_info['created_at']}",
            )
        ]

    elif name == "moderate_message":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.fetch_message(int(arguments["message_id"]))

        # Delete the message
        await message.delete(reason=arguments["reason"])

        # Handle timeout if specified
        if "timeout_minutes" in arguments and arguments["timeout_minutes"] > 0:
            if isinstance(message.author, discord.Member):
                duration = discord.utils.utcnow() + timedelta(
                    minutes=arguments["timeout_minutes"]
                )
                await message.author.timeout(duration, reason=arguments["reason"])
                return [
                    TextContent(
                        type="text",
                        text=f"Message deleted and user timed out for {arguments['timeout_minutes']} minutes.",
                    )
                ]

        return [TextContent(type="text", text="Message deleted successfully.")]

    # Server Information Tools
    elif name == "get_server_info":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))
        info = {
            "name": guild.name,
            "id": str(guild.id),
            "owner_id": str(guild.owner_id),
            "member_count": guild.member_count,
            "created_at": guild.created_at.isoformat(),
            "description": guild.description,
            "premium_tier": guild.premium_tier,
            "explicit_content_filter": str(guild.explicit_content_filter),
        }
        return [
            TextContent(
                type="text",
                text=f"Server Information:\n"
                + "\n".join(f"{k}: {v}" for k, v in info.items()),
            )
        ]

    elif name == "get_channels":
        try:
            guild = discord_client.get_guild(int(arguments["server_id"]))
            if guild:
                channel_list = []
                for channel in guild.channels:
                    channel_list.append(
                        f"#{channel.name} (ID: {channel.id}) - {channel.type}"
                    )

                return [
                    TextContent(
                        type="text",
                        text=f"Channels in {guild.name}:\n" + "\n".join(channel_list),
                    )
                ]
            else:
                return [TextContent(type="text", text="Guild not found")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "list_members":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))
        limit = min(int(arguments.get("limit", 100)), 1000)

        members = []
        async for member in guild.fetch_members(limit=limit):
            members.append(
                {
                    "id": str(member.id),
                    "name": member.name,
                    "nick": member.nick,
                    "joined_at": member.joined_at.isoformat()
                    if member.joined_at
                    else None,
                    "roles": [
                        str(role.id) for role in member.roles[1:]
                    ],  # Skip @everyone
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

    # Role Management Tools
    elif name == "add_role":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))
        member = await guild.fetch_member(int(arguments["user_id"]))
        role = guild.get_role(int(arguments["role_id"]))

        await member.add_roles(role, reason="Role added via MCP")
        return [
            TextContent(
                type="text", text=f"Added role {role.name} to user {member.name}"
            )
        ]

    elif name == "remove_role":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))
        member = await guild.fetch_member(int(arguments["user_id"]))
        role = guild.get_role(int(arguments["role_id"]))

        await member.remove_roles(role, reason="Role removed via MCP")
        return [
            TextContent(
                type="text", text=f"Removed role {role.name} from user {member.name}"
            )
        ]

    # Channel Management Tools
    elif name == "create_text_channel":
        guild = await discord_client.fetch_guild(int(arguments["server_id"]))
        category = None
        if "category_id" in arguments:
            category = guild.get_channel(int(arguments["category_id"]))

        channel = await guild.create_text_channel(
            name=arguments["name"],
            category=category,
            topic=arguments.get("topic"),
            reason="Channel created via MCP",
        )

        return [
            TextContent(
                type="text",
                text=f"Created text channel #{channel.name} (ID: {channel.id})",
            )
        ]

    elif name == "delete_channel":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        await channel.delete(reason=arguments.get("reason", "Channel deleted via MCP"))
        return [TextContent(type="text", text=f"Deleted channel successfully")]

    # Message Reaction Tools
    elif name == "add_reaction":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.fetch_message(int(arguments["message_id"]))
        await message.add_reaction(arguments["emoji"])
        return [
            TextContent(
                type="text", text=f"Added reaction {arguments['emoji']} to message"
            )
        ]

    elif name == "add_multiple_reactions":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.fetch_message(int(arguments["message_id"]))
        for emoji in arguments["emojis"]:
            await message.add_reaction(emoji)
        return [
            TextContent(
                type="text",
                text=f"Added reactions: {', '.join(arguments['emojis'])} to message",
            )
        ]

    elif name == "remove_reaction":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.fetch_message(int(arguments["message_id"]))
        await message.remove_reaction(arguments["emoji"], discord_client.user)
        return [
            TextContent(
                type="text", text=f"Removed reaction {arguments['emoji']} from message"
            )
        ]

    elif name == "list_servers":
        servers = []
        for guild in discord_client.guilds:
            servers.append(
                {
                    "id": str(guild.id),
                    "name": guild.name,
                    "member_count": guild.member_count,
                    "created_at": guild.created_at.isoformat(),
                }
            )

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

    raise ValueError(f"Unknown tool: {name}")


async def main():
    # Start Discord bot in the background
    asyncio.create_task(bot.start(DISCORD_TOKEN))

    # Run MCP server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
