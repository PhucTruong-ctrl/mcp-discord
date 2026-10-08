import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse

import aiohttp
import discord
from mcp.types import TextContent

from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token


def _json(payload: Dict[str, Any]) -> List[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def handle_download_attachment(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
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


async def handle_get_user_info(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = arguments.get("server_id")
    member = await deps["gateway"].resolve_member(arguments["user_id"], server_id)
    user = member.user if hasattr(member, "user") else member
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
            text="User information:\n"
            + f"Name: {user_info['name']}#{user_info['discriminator']}\n"
            + f"ID: {user_info['id']}\n"
            + f"Bot: {user_info['bot']}\n"
            + f"Created: {user_info['created_at']}",
        )
    ]


async def handle_moderate_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = arguments.get("server_id")
    reason = str(arguments.get("reason") or "")
    timeout_minutes = arguments.get("timeout_minutes")
    targets = {
        "message_id": str(arguments["message_id"]),
        "timeout_minutes": timeout_minutes,
        "reason": reason,
    }

    if bool(arguments.get("dry_run", True)):
        return _json(
            build_dry_run_result(
                "moderate_message",
                targets,
                {
                    "channel_id": str(arguments["channel_id"]),
                    "message_id": str(arguments["message_id"]),
                    "will_timeout": bool(timeout_minutes and timeout_minutes > 0),
                    "timeout_minutes": timeout_minutes,
                    "reason": reason,
                },
            )
        )
    verify_confirm_token("moderate_message", targets, arguments.get("confirm_token"))

    channel = await deps["gateway"].resolve_text_or_thread_channel(
        arguments["channel_id"], server_id
    )
    message = await channel.fetch_message(int(arguments["message_id"]))

    await message.delete(reason=reason)

    if timeout_minutes and timeout_minutes > 0:
        if isinstance(message.author, discord.Member):
            duration = discord.utils.utcnow() + timedelta(minutes=timeout_minutes)
            await message.author.timeout(duration, reason=reason)
            return [
                TextContent(
                    type="text",
                    text=f"Message deleted and user timed out for {timeout_minutes} minutes.",
                )
            ]

    return [TextContent(type="text", text="Message deleted successfully.")]


async def handle_add_reaction(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = arguments.get("server_id")
    channel = await deps["gateway"].resolve_text_or_thread_channel(
        arguments["channel_id"], server_id
    )
    message = await channel.fetch_message(int(arguments["message_id"]))
    await message.add_reaction(arguments["emoji"])
    return [
        TextContent(type="text", text=f"Added reaction {arguments['emoji']} to message")
    ]


async def handle_add_multiple_reactions(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = arguments.get("server_id")
    channel = await deps["gateway"].resolve_text_or_thread_channel(
        arguments["channel_id"], server_id
    )
    message = await channel.fetch_message(int(arguments["message_id"]))
    for emoji in arguments["emojis"]:
        await message.add_reaction(emoji)
    return [
        TextContent(
            type="text",
            text=f"Added reactions: {', '.join(arguments['emojis'])} to message",
        )
    ]


async def handle_remove_reaction(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = arguments.get("server_id")
    reason = str(arguments.get("reason") or "")
    targets = {
        "channel_id": str(arguments["channel_id"]),
        "message_id": str(arguments["message_id"]),
        "emoji": str(arguments["emoji"]),
        "reason": reason,
    }

    if bool(arguments.get("dry_run", True)):
        return _json(
            build_dry_run_result(
                "remove_reaction",
                targets,
                {
                    "channel_id": str(arguments["channel_id"]),
                    "message_id": str(arguments["message_id"]),
                    "emoji": str(arguments["emoji"]),
                    "reason": reason,
                },
            )
        )
    verify_confirm_token("remove_reaction", targets, arguments.get("confirm_token"))

    channel = await deps["gateway"].resolve_text_or_thread_channel(
        arguments["channel_id"], server_id
    )
    message = await channel.fetch_message(int(arguments["message_id"]))
    await message.remove_reaction(arguments["emoji"], deps["discord_client"].user)
    return [
        TextContent(
            type="text", text=f"Removed reaction {arguments['emoji']} from message"
        )
    ]
