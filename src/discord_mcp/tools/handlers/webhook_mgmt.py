"""Webhook-management handlers: fetch/edit/delete webhooks and their messages.

Every payload here is built so the raw webhook token never appears: URLs and
token fields are masked to the last 4 characters, and the confirm-token gate
binds on ``webhook_id`` alone.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import discord
from mcp.types import TextContent

from discord_mcp.core.common import as_id, json_text, require_gateway
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.validation import require_reason, validate_snowflake

_WEBHOOK_URL = "https://discord.com/api/webhooks"


def masked_token(token: str) -> str:
    """Last-4 mask for a webhook token; never returns the raw token."""
    value = str(token or "")
    return "****" + value[-4:] if len(value) > 4 else "****"


def _webhook_row(webhook: Any, token: str) -> Dict[str, Any]:
    webhook_id = as_id(getattr(webhook, "id", None))
    avatar = getattr(webhook, "avatar", None)
    webhook_type = getattr(webhook, "type", None)
    created_at = getattr(webhook, "created_at", None)
    masked = masked_token(token)
    return {
        "id": webhook_id,
        "name": getattr(webhook, "name", None),
        "type": (
            str(getattr(webhook_type, "name", webhook_type)).lower()
            if webhook_type is not None
            else None
        ),
        "channelId": as_id(getattr(webhook, "channel_id", None)),
        "guildId": as_id(getattr(webhook, "guild_id", None)),
        "applicationId": as_id(getattr(webhook, "application_id", None)),
        "avatarUrl": str(avatar.url) if avatar is not None else None,
        "createdAt": (
            created_at.isoformat() if hasattr(created_at, "isoformat") else None
        ),
        "url": f"{_WEBHOOK_URL}/{webhook_id}/{masked}",
        "tokenMasked": masked,
    }


def _message_row(message: Any) -> Dict[str, Any]:
    channel = getattr(message, "channel", None)
    channel_id = getattr(channel, "id", None)
    if channel_id is None:
        channel_id = getattr(message, "channel_id", None)
    author = getattr(message, "author", None)
    created_at = getattr(message, "created_at", None)
    embeds = [embed.to_dict() for embed in getattr(message, "embeds", None) or []]
    attachments = [
        {"filename": a.filename, "size": a.size, "url": str(a.url)}
        for a in getattr(message, "attachments", None) or []
    ]
    return {
        "messageId": as_id(getattr(message, "id", None)),
        "channelId": as_id(channel_id),
        "content": getattr(message, "content", None),
        "authorId": as_id(getattr(author, "id", None)),
        "createdAt": (
            created_at.isoformat() if hasattr(created_at, "isoformat") else None
        ),
        "embeds": embeds,
        "attachments": attachments,
    }


def _is_dry_run(arguments: Dict[str, Any]) -> bool:
    return bool(arguments.get("dry_run", True))


def _snowflake(arguments: Dict[str, Any], key: str, action: str) -> int:
    value = arguments.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{key} is required for {action}")
    return validate_snowflake(str(value))


def _require_token(arguments: Dict[str, Any], action: str) -> str:
    token = arguments.get("webhook_token")
    if token is None or str(token).strip() == "":
        raise ValueError(f"webhook_token is required for {action}")
    return str(token)


def _check_avatar_url(value: Any) -> str:
    url = str(value)
    if urlparse(url).scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"avatar_url: only http/https URLs are supported, got '{url}'"
        )
    return url


def _download_avatar(url: str) -> bytes:
    try:
        with urlopen(url, timeout=30) as response:  # noqa: S310 - scheme checked above
            return response.read()
    except (URLError, OSError) as exc:
        raise ValueError(f"avatar_url: failed to download '{url}': {exc}") from exc


async def _fetch_webhook(gateway: Any, webhook_id: int, token: str) -> Any:
    # Reuses the gateway lookup: it translates discord.NotFound into a
    # ValueError naming the webhook id.
    return await gateway.fetch_webhook(str(webhook_id), token)


async def _fetch_message(webhook: Any, webhook_id: int, message_id: int) -> Any:
    try:
        return await webhook.fetch_message(message_id)
    except discord.NotFound:
        raise ValueError(
            f"Webhook message '{message_id}' not found for webhook '{webhook_id}'"
        ) from None


async def handle_get_webhook(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_webhook")
    webhook_id = _snowflake(arguments, "webhook_id", "get_webhook")
    token = _require_token(arguments, "get_webhook")
    webhook = await _fetch_webhook(gateway, webhook_id, token)
    return json_text(_webhook_row(webhook, token))


async def handle_edit_webhook(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_webhook")
    webhook_id = _snowflake(arguments, "webhook_id", "edit_webhook")
    token = _require_token(arguments, "edit_webhook")
    reason = arguments.get("reason")

    kwargs: Dict[str, Any] = {}
    planned: List[str] = []
    name = arguments.get("name")
    if name is not None:
        kwargs["name"] = str(name)
        planned.append("name")
    channel_id = arguments.get("channel_id")
    if channel_id is not None:
        kwargs["channel"] = discord.Object(
            id=_snowflake(arguments, "channel_id", "edit_webhook")
        )
        planned.append("channel_id")
    avatar_url = arguments.get("avatar_url")
    if avatar_url is not None:
        avatar_url = _check_avatar_url(avatar_url)
        planned.append("avatar_url")
    if not planned:
        raise ValueError(
            "edit_webhook requires at least one of: name, channel_id, avatar_url"
        )

    webhook = await _fetch_webhook(gateway, webhook_id, token)

    action = "edit_webhook"
    targets = {"webhook_id": str(webhook_id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action, targets, {"webhookId": str(webhook.id), "updates": sorted(planned)}
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    if avatar_url is not None:
        kwargs["avatar"] = await asyncio.to_thread(_download_avatar, avatar_url)
    if reason is not None:
        kwargs["reason"] = reason
    edited = await webhook.edit(**kwargs)
    return json_text(
        {"status": "executed", "action": action, "webhook": _webhook_row(edited, token)}
    )


async def handle_delete_webhook(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_webhook")
    webhook_id = _snowflake(arguments, "webhook_id", "delete_webhook")
    token = _require_token(arguments, "delete_webhook")
    action = "delete_webhook"
    reason = require_reason(arguments.get("reason"), action)

    webhook = await _fetch_webhook(gateway, webhook_id, token)

    targets = {"webhook_id": str(webhook_id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {"webhookId": str(webhook.id), "name": webhook.name},
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await webhook.delete(reason=reason)
    return json_text({"status": "executed", "action": action, "webhookId": str(webhook_id)})


async def handle_get_webhook_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_webhook_message")
    webhook_id = _snowflake(arguments, "webhook_id", "get_webhook_message")
    message_id = _snowflake(arguments, "message_id", "get_webhook_message")
    token = _require_token(arguments, "get_webhook_message")

    webhook = await _fetch_webhook(gateway, webhook_id, token)
    message = await _fetch_message(webhook, webhook_id, message_id)
    return json_text(_message_row(message))


async def handle_edit_webhook_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_webhook_message")
    webhook_id = _snowflake(arguments, "webhook_id", "edit_webhook_message")
    message_id = _snowflake(arguments, "message_id", "edit_webhook_message")
    token = _require_token(arguments, "edit_webhook_message")
    content = arguments.get("content")
    if content is None or str(content) == "":
        raise ValueError(
            "content is required for edit_webhook_message; "
            "an empty edit is a no-op"
        )
    content = str(content)

    webhook = await _fetch_webhook(gateway, webhook_id, token)
    message = await _fetch_message(webhook, webhook_id, message_id)

    action = "edit_webhook_message"
    targets = {"webhook_id": str(webhook_id)}
    if _is_dry_run(arguments):
        row = _message_row(message)
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {
                    "messageId": row["messageId"],
                    "channelId": row["channelId"],
                    "content": content,
                },
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    edited = await webhook.edit_message(message_id, content=content)
    row = _message_row(edited)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "messageId": row["messageId"],
            "channelId": row["channelId"],
        }
    )


async def handle_delete_webhook_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_webhook_message")
    webhook_id = _snowflake(arguments, "webhook_id", "delete_webhook_message")
    message_id = _snowflake(arguments, "message_id", "delete_webhook_message")
    token = _require_token(arguments, "delete_webhook_message")
    action = "delete_webhook_message"
    # Discord's webhook-message delete endpoint accepts no audit-log reason,
    # so the reason is validated for the gate but never transmitted.
    require_reason(arguments.get("reason"), action)

    webhook = await _fetch_webhook(gateway, webhook_id, token)
    message = await _fetch_message(webhook, webhook_id, message_id)

    targets = {"webhook_id": str(webhook_id)}
    if _is_dry_run(arguments):
        row = _message_row(message)
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {
                    "messageId": row["messageId"],
                    "channelId": row["channelId"],
                    "content": row["content"],
                },
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await webhook.delete_message(message_id)
    return json_text(
        {"status": "executed", "action": action, "messageId": str(message_id)}
    )
