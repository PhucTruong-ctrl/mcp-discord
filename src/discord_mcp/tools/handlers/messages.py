import json
from typing import Any, Dict, List

import discord
from mcp.types import TextContent

from discord_mcp.core.serialize import _serialize_embed


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

    gateway = deps["gateway"]
    channel = await gateway.resolve_text_or_thread_channel(
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

    gateway = deps["gateway"]
    try_int = deps["try_int"]
    channel = await gateway.resolve_text_or_thread_channel(
        str(channel_identifier), server_id
    )
    limit = min(int(arguments.get("limit", 10)), 100)
    before = arguments.get("before")
    before_obj = discord.Object(id=int(before)) if try_int(before) else None
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
                "embeds": [_serialize_embed(embed) for embed in message.embeds],
            }
        )

    def format_reaction(r):
        return f"{r['emoji']}({r['count']})"

    def format_embed(e):
        parts = []
        if e.get("title"):
            parts.append(f"  title: {e['title']}")
        if e.get("description"):
            parts.append(f"  description: {e['description']}")
        if e.get("url"):
            parts.append(f"  url: {e['url']}")
        if e.get("image"):
            parts.append(f"  image: {e['image']}")
        if e.get("thumbnail"):
            parts.append(f"  thumbnail: {e['thumbnail']}")
        if e.get("author"):
            parts.append(f"  author: {e['author']}")
        if e.get("footer"):
            parts.append(f"  footer: {e['footer']}")
        if e.get("timestamp"):
            parts.append(f"  timestamp: {e['timestamp']}")
        if e.get("color") is not None:
            parts.append(f"  color: {e['color']}")
        if e.get("fields"):
            for field in e["fields"]:
                parts.append(
                    f"  field: {field.get('name')} = {field.get('value')}"
                    + (
                        f" (inline={field.get('inline')})"
                        if field.get("inline") is not None
                        else ""
                    )
                )
        return "\n".join(parts)

    embed_json = json.dumps(
        [
            {"id": m["id"], "author": m["author"], "embeds": m["embeds"]}
            for m in messages
            if m["embeds"]
        ],
        ensure_ascii=False,
        indent=2,
    )

    return [
        TextContent(
            type="text",
            text=f"Retrieved {len(messages)} messages:\n\n"
            + "\n\n".join(
                [
                    f"{m['author']} ({m['timestamp']}) [message_id={m['id']}]: {m['content']}\n"
                    + f"Reactions: {', '.join([format_reaction(r) for r in m['reactions']]) if m['reactions'] else 'No reactions'}"
                    + (
                        "\nEmbeds:\n"
                        + "\n".join([format_embed(e) for e in m["embeds"]])
                        if m["embeds"]
                        else ""
                    )
                    for m in messages
                ]
            ),
        ),
        TextContent(
            type="text",
            text=f"Embed data: {embed_json}"
            if embed_json != "[]"
            else "No embeds found",
        ),
    ]


async def handle_edit_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    channel = await gateway.fetch_channel(arguments["channel_id"])
    message = await channel.fetch_message(int(arguments["message_id"]))
    await message.edit(content=arguments["content"])
    return [
        TextContent(
            type="text", text=f"Message edited successfully. Message ID: {message.id}"
        )
    ]


async def handle_reply_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = arguments.get("server_id") or arguments.get("server")
    channel_identifier = arguments.get("channel_id") or arguments.get("channel")
    content = arguments.get("content")
    message_id = arguments.get("message_id")

    if not channel_identifier:
        raise ValueError("channel_id (or channel) is required")
    if not message_id:
        raise ValueError("message_id is required")
    if content is None:
        raise ValueError("content is required")

    gateway = deps["gateway"]
    channel = await gateway.resolve_text_or_thread_channel(
        str(channel_identifier), server_id
    )
    target = await channel.fetch_message(int(message_id))
    reply = await channel.send(content=str(content), reference=target)
    return [
        TextContent(
            type="text",
            text=f"Reply sent successfully. Reply Message ID: {reply.id}, in reply to Message ID: {message_id}",
        )
    ]
