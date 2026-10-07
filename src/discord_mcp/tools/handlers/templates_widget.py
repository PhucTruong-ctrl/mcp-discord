"""Template and widget handlers: guild templates (account-level) and the widget."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from mcp.types import TextContent

from discord_mcp.core.common import as_id, channel_row, json_text, require_gateway
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.validation import require_reason, validate_snowflake
from discord_mcp.tools.handlers.emoji import _emoji_row, _sticker_row

# ``discord.gg/<code>`` and ``discord.com/invite/<code>`` are invite-style links; a
# template code is also accepted bare. discord.py's own ``utils.resolve_template`` only
# knows ``discord.new`` / ``discord.com/template``, so the prefixes are stripped here.
_URL_PREFIX_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?"
    r"(?:discord\.gg|discord(?:app)?\.com/(?:invite|template)|discord\.new)"
    r"/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_CODE_RE = re.compile(r"[A-Za-z0-9_-]+")


def _is_dry_run(arguments: Dict[str, Any]) -> bool:
    return bool(arguments.get("dry_run", True))


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value else None


def _enum_name(value: Any) -> Any:
    return getattr(value, "name", value) or value


def _client(deps: Dict[str, Any], tool: str) -> Any:
    client = deps.get("discord_client")
    if not client:
        raise ValueError(f"discord_client is required for {tool}")
    return client


def _code(arguments: Dict[str, Any], tool: str) -> str:
    """Template code, from a bare code or an invite-style URL."""
    raw = str(arguments.get("code") or "").strip()
    if not raw:
        raise ValueError(f"code is required for {tool}")
    match = _URL_PREFIX_RE.match(raw)
    if match:
        return match.group(1)
    if _CODE_RE.fullmatch(raw):
        return raw
    raise ValueError(
        f"code must be a template code or a discord.gg/<code> URL (got {raw!r})"
    )


def _template_row(template: Any) -> Dict[str, Any]:
    return {
        "code": template.code,
        "name": template.name,
        "description": template.description,
        "usageCount": template.uses,
        "creatorId": as_id(getattr(template.creator, "id", None)),
        "createdAt": _iso(template.created_at),
        "updatedAt": _iso(template.updated_at),
    }


def _serialized_guild(template: Any) -> Dict[str, Any]:
    """The guild snapshot the template holds (``serialized_source_guild``)."""
    guild = getattr(template, "source_guild", None)
    if guild is None:
        return {}
    return {
        "id": as_id(guild.id),
        "name": guild.name,
        "description": guild.description,
        "features": sorted(str(feature) for feature in (guild.features or [])),
        "verificationLevel": _enum_name(guild.verification_level),
        "defaultNotifications": _enum_name(guild.default_notifications),
        "explicitContentFilter": _enum_name(guild.explicit_content_filter),
        "afkTimeout": guild.afk_timeout,
        "roles": sorted(
            [{"id": as_id(role.id), "name": role.name} for role in guild.roles],
            key=lambda role: role["id"],
        ),
        "channels": sorted(
            (channel_row(channel) for channel in guild.channels),
            key=lambda channel: channel["id"],
        ),
    }


async def handle_list_templates(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "list_templates")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))
    templates = await guild.templates()
    rows = sorted(
        (_template_row(template) for template in templates),
        key=lambda row: row["code"],
    )
    return json_text(
        {"serverId": str(guild.id), "count": len(rows), "templates": rows}
    )


async def handle_create_template(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_template")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))

    name = str(arguments.get("name") or "").strip()
    if not name:
        raise ValueError("name is required for create_template")
    description = arguments.get("description")
    # guild.create_template takes no audit-log reason, so a given reason is recorded
    # here and never transmitted to Discord.
    reason = str(arguments.get("reason") or "").strip() or None

    action = "create_template"
    targets = {"server_id": str(guild.id)}
    details: Dict[str, Any] = {"name": name}
    if description is not None:
        details["description"] = str(description)
    if reason is not None:
        details["reason"] = reason
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs: Dict[str, Any] = {"name": name}
    if description is not None:
        kwargs["description"] = str(description)
    template = await guild.create_template(**kwargs)
    payload: Dict[str, Any] = {
        "status": "executed",
        "action": action,
        "code": template.code,
        "name": template.name,
        "url": template.url,
    }
    if reason is not None:
        payload["reason"] = reason
    return json_text(payload)


async def handle_get_template(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_template")
    client = _client(deps, "get_template")
    code = _code(arguments, "get_template")
    template = await client.fetch_template(code)
    return json_text(
        {
            **_template_row(template),
            "serializedGuild": _serialized_guild(template),
        }
    )


async def handle_sync_template(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "sync_template")
    client = _client(deps, "sync_template")
    code = _code(arguments, "sync_template")
    template = await client.fetch_template(code)
    # discord.py takes MISSING (not None) for unset template fields.
    updates: Dict[str, Any] = {}
    if arguments.get("name") is not None:
        updates["name"] = str(arguments["name"])
    if arguments.get("description") is not None:
        updates["description"] = str(arguments["description"])

    action = "sync_template"
    targets = {"code": code}
    details = dict(updates)
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    synced = await template.sync()
    if updates:
        synced = await synced.edit(**updates)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "code": synced.code,
            "name": synced.name,
            "description": synced.description,
        }
    )


async def handle_edit_template(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_template")
    client = _client(deps, "edit_template")
    code = _code(arguments, "edit_template")
    updates: Dict[str, Any] = {}
    if arguments.get("name") is not None:
        updates["name"] = str(arguments["name"])
    if arguments.get("description") is not None:
        updates["description"] = str(arguments["description"])
    if not updates:
        raise ValueError("edit_template requires at least one of: name, description")

    template = await client.fetch_template(code)
    action = "edit_template"
    targets = {"code": code}
    details = dict(updates)
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    edited = await template.edit(**updates)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "code": edited.code,
            "name": edited.name,
            "description": edited.description,
        }
    )


async def handle_delete_template(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_template")
    client = _client(deps, "delete_template")
    code = _code(arguments, "delete_template")
    action = "delete_template"
    reason = require_reason(arguments.get("reason"), action)
    template = await client.fetch_template(code)

    targets = {"code": code}
    details = {"name": template.name, "reason": reason}
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    # Discord's template delete endpoint accepts no audit-log reason, so the reason is
    # validated for the gate but never transmitted.
    await template.delete()
    return json_text({"status": "executed", "action": action, "code": code})


async def handle_get_guild_preview(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_guild_preview")
    client = _client(deps, "get_guild_preview")
    preview = await client.fetch_guild_preview(
        validate_snowflake(str(arguments["server_id"]))
    )

    def asset_url(asset: Any) -> Optional[str]:
        return str(asset.url) if asset is not None else None

    return json_text(
        {
            "serverId": str(preview.id),
            "name": preview.name,
            "description": preview.description,
            "iconUrl": asset_url(preview.icon),
            "splashUrl": asset_url(preview.splash),
            "discoverySplashUrl": asset_url(preview.discovery_splash),
            "emojis": [_emoji_row(emoji) for emoji in preview.emojis],
            "stickers": [_sticker_row(sticker) for sticker in preview.stickers],
            "features": sorted(str(feature) for feature in (preview.features or [])),
            "approximateMemberCount": preview.approximate_member_count,
            "approximatePresenceCount": preview.approximate_presence_count,
            "createdAt": _iso(preview.created_at),
        }
    )


async def handle_get_widget_settings(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_widget_settings")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))

    # Widget carries no enabled flag (and is refused by Discord while the widget is
    # off), so the flag is read off the guild.
    enabled = bool(guild.widget_enabled)
    widget_channel = getattr(guild, "widget_channel", None)
    channel_id = as_id(getattr(widget_channel, "id", None)) or as_id(
        getattr(guild, "_widget_channel_id", None)
    )
    if enabled:
        widget = await guild.widget()
        return json_text(
            {
                "serverId": str(guild.id),
                "enabled": True,
                "name": widget.name,
                "channelId": channel_id,
                "inviteUrl": widget.invite_url,
                "jsonUrl": widget.json_url,
                "presenceCount": widget.presence_count,
            }
        )
    return json_text(
        {
            "serverId": str(guild.id),
            "enabled": False,
            "name": guild.name,
            "channelId": channel_id,
            "inviteUrl": None,
            "jsonUrl": None,
            "presenceCount": None,
        }
    )


async def handle_edit_widget_settings(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_widget_settings")
    guild = await gateway.resolve_guild(str(arguments["server_id"]))

    enabled = arguments.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError(f"enabled must be a boolean (got {enabled!r})")
    channel_id = str(arguments.get("channel_id") or "").strip()
    if enabled is None and not channel_id:
        raise ValueError(
            "edit_widget_settings requires at least one of: enabled, channel_id"
        )

    channel = None
    if channel_id:
        channel = guild.get_channel(validate_snowflake(channel_id))
        if channel is None:
            raise ValueError(
                f"Channel '{channel_id}' not found in server '{guild.name}'"
            )
    reason = arguments.get("reason")

    action = "edit_widget_settings"
    targets = {"server_id": str(guild.id)}
    if channel_id:
        targets["channel_id"] = channel_id
    details: Dict[str, Any] = {}
    if enabled is not None:
        details["enabled"] = enabled
    if channel_id:
        details["channelId"] = channel_id
    if reason is not None:
        details["reason"] = str(reason)
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    # MISSING (not None) means "leave alone", so unset fields are omitted entirely.
    kwargs: Dict[str, Any] = {}
    if enabled is not None:
        kwargs["enabled"] = enabled
    if channel is not None:
        kwargs["channel"] = channel
    if reason is not None:
        kwargs["reason"] = str(reason)
    await guild.edit_widget(**kwargs)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "serverId": str(guild.id),
            "enabled": enabled,
            "channelId": channel_id or None,
        }
    )
