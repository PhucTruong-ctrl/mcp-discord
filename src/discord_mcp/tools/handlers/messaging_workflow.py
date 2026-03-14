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
    await webhook.send(content=arguments["content"], username=arguments.get("username"))
    payload = {
        "executed": True,
        "webhookId": str(arguments["webhook_id"]),
    }
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
