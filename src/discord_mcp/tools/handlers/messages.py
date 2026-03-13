from typing import Any, Dict, List

import discord
from mcp.types import TextContent


async def handle_send_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = arguments.get("server_id") or arguments.get("server")
    channel_identifier = arguments.get("channel_id") or arguments.get("channel")
    content = arguments.get("content") or arguments.get("message")

    if not channel_identifier:
        raise ValueError("channel_id (or channel) is required")
    if content is None:
        raise ValueError("content (or message) is required")

    channel = await deps["resolve_text_or_thread_channel"](
        str(channel_identifier), server_id
    )
    message = await channel.send(str(content))
    return [
        TextContent(
            type="text", text=f"Message sent successfully. Message ID: {message.id}"
        )
    ]


async def handle_read_messages(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = arguments.get("server_id") or arguments.get("server")
    channel_identifier = arguments.get("channel_id") or arguments.get("channel")
    if not channel_identifier:
        raise ValueError("channel_id (or channel) is required")

    channel = await deps["resolve_text_or_thread_channel"](
        str(channel_identifier), server_id
    )
    limit = min(int(arguments.get("limit", 10)), 100)
    before = arguments.get("before")
    before_obj = discord.Object(id=int(before)) if deps["try_int"](before) else None
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
            reaction_data.append({"emoji": emoji_str, "count": reaction.count})
        messages.append(
            {
                "id": str(message.id),
                "author": str(message.author),
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
                "reactions": reaction_data,
            }
        )

    def format_reaction(r: Dict[str, Any]) -> str:
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


async def handle_edit_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    channel = await deps["discord_client"].fetch_channel(int(arguments["channel_id"]))
    message = await channel.fetch_message(int(arguments["message_id"]))
    await message.edit(content=arguments["content"])
    return [
        TextContent(
            type="text", text=f"Message edited successfully. Message ID: {message.id}"
        )
    ]
