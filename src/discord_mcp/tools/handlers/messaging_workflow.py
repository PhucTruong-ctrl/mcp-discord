import json
from typing import Any, Dict, List

import discord
from mcp.types import TextContent


def _build_embed(embed_payload: Dict[str, Any]) -> discord.Embed:
    embed = discord.Embed(
        title=embed_payload.get("title"),
        description=embed_payload.get("description"),
        color=embed_payload.get("color"),
    )
    for field in embed_payload.get("fields", []):
        embed.add_field(
            name=str(field.get("name", "")),
            value=str(field.get("value", "")),
            inline=bool(field.get("inline", False)),
        )
    return embed


async def handle_send_embed_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments.get("server_id") or arguments.get("server")
    channel = await gateway.resolve_text_or_thread_channel(
        arguments["channel_id"], server_id
    )
    embed = _build_embed(arguments["embed"])
    message = await channel.send(content=arguments.get("content"), embed=embed)
    return [
        TextContent(
            type="text",
            text=f"Embed message sent successfully. Message ID: {message.id}",
        )
    ]


async def handle_send_rich_announcement(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    payload = {
        "title": arguments["title"],
        "description": arguments["body"],
        "color": arguments.get("color"),
    }
    return await handle_send_embed_message(
        {
            "server_id": arguments.get("server_id") or arguments.get("server"),
            "channel_id": arguments["channel_id"],
            "embed": payload,
        },
        deps,
    )


async def handle_crosspost_announcement(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    channel = await deps["gateway"].fetch_channel(arguments["channel_id"])
    message = await channel.fetch_message(int(arguments["message_id"]))
    await message.crosspost()
    return [
        TextContent(
            type="text",
            text=f"Announcement message {arguments['message_id']} crossposted",
        )
    ]


async def handle_create_channel_webhook(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    channel = await deps["gateway"].fetch_channel(arguments["channel_id"])
    webhook = await channel.create_webhook(
        name=arguments["name"], reason=arguments.get("reason")
    )
    payload = {
        "webhookId": str(webhook.id),
        "name": webhook.name,
        "channelId": str(channel.id),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_list_channel_webhooks(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    channel = await deps["gateway"].fetch_channel(arguments["channel_id"])
    webhooks = await channel.webhooks()
    payload = {
        "channelId": str(channel.id),
        "webhooks": [
            {
                "id": str(webhook.id),
                "name": webhook.name,
                "tokenPresent": bool(getattr(webhook, "token", None)),
            }
            for webhook in webhooks
        ],
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_execute_channel_webhook(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    webhook = await deps["gateway"].fetch_webhook(
        arguments["webhook_id"], arguments["token"]
    )
    content = arguments.get("content")
    embed_payload = arguments.get("embed")
    file_urls = arguments.get("file_urls")
    components = arguments.get("components")
    if content is None and embed_payload is None and not file_urls and not components:
        raise ValueError("content, embed, file_urls, or components is required")

    kwargs: Dict[str, Any] = {}
    if content is not None:
        kwargs["content"] = str(content)
    if arguments.get("username") is not None:
        kwargs["username"] = arguments["username"]
    if arguments.get("avatar_url") is not None:
        kwargs["avatar_url"] = arguments["avatar_url"]
    if arguments.get("tts") is not None:
        kwargs["tts"] = bool(arguments["tts"])
    if embed_payload is not None:
        kwargs["embed"] = _build_embed(embed_payload)
    if file_urls:
        from discord_mcp.core.common import fetch_bytes
        import io
        files = []
        for url in file_urls:
            data = await fetch_bytes(str(url), "file_urls")
            from urllib.parse import urlparse
            import os
            filename = os.path.basename(urlparse(str(url)).path) or "download"
            files.append(discord.File(io.BytesIO(data), filename=filename))
        kwargs["files"] = files
    if components:
        from discord_mcp.tools.handlers.messages_advanced import _build_view
        kwargs["view"] = _build_view(components)
    if arguments.get("wait") is not None:
        kwargs["wait"] = bool(arguments["wait"])

    message = await webhook.send(**kwargs)
    payload: Dict[str, Any] = {
        "executed": True,
        "webhookId": str(arguments["webhook_id"]),
    }
    if message is not None:
        from discord_mcp.core.serialize import _serialize_message
        payload["message"] = _serialize_message(message)
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_list_guild_integrations(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(
        arguments.get("server_id") or arguments.get("server")
    )
    integrations = await guild.integrations()
    payload = {
        "serverId": str(guild.id),
        "integrations": [
            {
                "id": str(integration.id),
                "name": integration.name,
                "type": integration.type,
                "enabled": integration.enabled,
            }
            for integration in integrations
        ],
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_get_guild_vanity_url(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(
        arguments.get("server_id") or arguments.get("server")
    )
    code = getattr(guild, "vanity_url_code", None)
    payload = {
        "serverId": str(guild.id),
        "serverName": guild.name,
        "vanityCode": code,
        "vanityUrl": f"https://discord.gg/{code}" if code else None,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]
