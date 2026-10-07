"""Monetization (SKUs/entitlements) and application-command tools.

These tools talk to the application itself via ``deps["discord_client"]`` -
no guild is involved - but they still require a live gateway so the MCP server
never answers from a half-connected client.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import discord
from mcp.types import TextContent

from discord_mcp.core.common import as_id, json_text, require_gateway
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.validation import (
    require_reason,
    validate_enum,
    validate_limit,
    validate_snowflake,
)

_OWNER_TYPES = {
    "user": discord.EntitlementOwnerType.user,
    "guild": discord.EntitlementOwnerType.guild,
}


def _is_dry_run(arguments: Dict[str, Any]) -> bool:
    return bool(arguments.get("dry_run", True))


def _client(deps: Dict[str, Any], tool: str) -> Any:
    client = deps.get("discord_client")
    if not client:
        raise ValueError(f"discord_client is required for {tool}")
    return client



def _tree(deps: Dict[str, Any], tool: str) -> Any:
    """Return the application command tree, or explain why it is unavailable.

    ``Client.tree`` only exists on ``discord.ext.commands.Bot``; a bare
    ``discord.Client`` carries no command tree, so say so plainly rather than
    letting an AttributeError escape from the call site.
    """
    client = _client(deps, tool)
    tree = getattr(client, "tree", None)
    if tree is None:
        raise ValueError(
            f"{tool} needs a bot with an application command tree "
            "(discord.ext.commands.Bot); this client has none"
        )
    return tree


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def _iso(value: Any) -> Optional[str]:
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else None


def _owner_type(value: Any) -> discord.EntitlementOwnerType:
    key = validate_enum(str(value), list(_OWNER_TYPES), "owner_type")
    return _OWNER_TYPES[key]


def _guild_scope(value: Any) -> Tuple[Optional[discord.Object], Optional[str]]:
    """Map an optional guild_id to (Snowflake for discord.py, string for payload)."""
    if value is None or value == "":
        return None, None
    guild_id = validate_snowflake(str(value))
    return discord.Object(id=guild_id), str(guild_id)


def _sku_row(sku: Any) -> Dict[str, Any]:
    return {
        "id": as_id(getattr(sku, "id", None)),
        "name": getattr(sku, "name", None),
        "slug": getattr(sku, "slug", None),
        "type": _enum_name(getattr(sku, "type", None)),
    }


def _entitlement_row(entitlement: Any) -> Dict[str, Any]:
    return {
        "id": as_id(getattr(entitlement, "id", None)),
        "skuId": as_id(getattr(entitlement, "sku_id", None)),
        "userId": as_id(getattr(entitlement, "user_id", None)),
        "guildId": as_id(getattr(entitlement, "guild_id", None)),
        "type": _enum_name(getattr(entitlement, "type", None)),
        "startsAt": _iso(getattr(entitlement, "starts_at", None)),
        "endsAt": _iso(getattr(entitlement, "ends_at", None)),
        "consumed": bool(getattr(entitlement, "consumed", False)),
        "deleted": bool(getattr(entitlement, "deleted", False)),
    }


def _option_row(option: Any) -> Dict[str, Any]:
    row = {
        "name": getattr(option, "name", None),
        "description": getattr(option, "description", None),
        "type": _enum_name(getattr(option, "type", None)),
    }
    required = getattr(option, "required", None)
    if required is not None:
        row["required"] = bool(required)
    nested = getattr(option, "options", None)
    if nested:
        row["options"] = [_option_row(child) for child in nested]
    return row


def _command_row(command: Any) -> Dict[str, Any]:
    return {
        "id": as_id(getattr(command, "id", None)),
        "name": getattr(command, "name", None),
        "description": getattr(command, "description", None),
        "type": _enum_name(getattr(command, "type", None)),
        "options": [
            _option_row(option)
            for option in (getattr(command, "options", None) or [])
        ],
    }


async def _fetch_entitlement(client: Any, raw: str, entitlement_id: int) -> Any:
    try:
        return await client.fetch_entitlement(entitlement_id)
    except discord.NotFound:
        raise ValueError(f"Entitlement '{raw}' not found") from None


async def handle_list_skus(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    require_gateway(deps, "list_skus")
    client = _client(deps, "list_skus")

    skus = await client.fetch_skus()
    rows = [_sku_row(sku) for sku in skus]
    return json_text({"monetized": bool(rows), "count": len(rows), "skus": rows})


async def handle_list_entitlements(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    require_gateway(deps, "list_entitlements")
    client = _client(deps, "list_entitlements")

    limit = validate_limit(arguments.get("limit"), 100, 1000)
    raw_sku_ids = arguments.get("sku_ids") or []
    if isinstance(raw_sku_ids, str):
        raw_sku_ids = [raw_sku_ids]
    sku_ids = [discord.Object(id=validate_snowflake(str(v))) for v in raw_sku_ids]
    user_id = arguments.get("user_id")
    guild_id = arguments.get("guild_id")

    rows = [
        _entitlement_row(entitlement)
        async for entitlement in client.entitlements(
            limit=limit,
            skus=sku_ids or None,
            user=(
                discord.Object(id=validate_snowflake(str(user_id)))
                if user_id
                else None
            ),
            guild=(
                discord.Object(id=validate_snowflake(str(guild_id)))
                if guild_id
                else None
            ),
            exclude_ended=bool(arguments.get("exclude_ended", False)),
            exclude_deleted=bool(arguments.get("exclude_deleted", True)),
        )
    ]
    return json_text({"count": len(rows), "entitlements": rows})


async def handle_get_entitlement(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    require_gateway(deps, "get_entitlement")
    client = _client(deps, "get_entitlement")

    raw = str(arguments["entitlement_id"])
    entitlement = await _fetch_entitlement(client, raw, validate_snowflake(raw))
    return json_text(_entitlement_row(entitlement))


async def handle_create_entitlement(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    require_gateway(deps, "create_entitlement")
    client = _client(deps, "create_entitlement")

    raw_sku = str(arguments["sku_id"])
    sku_id = validate_snowflake(raw_sku)
    raw_owner = str(arguments["owner_id"])
    owner_id = validate_snowflake(raw_owner)
    owner_type = _owner_type(arguments.get("owner_type"))

    action = "create_entitlement"
    # Discord's create-entitlement endpoint accepts no audit-log reason, so the
    # reason is validated for the gate but never transmitted.
    reason = require_reason(arguments.get("reason"), action)
    targets = {
        "sku_id": str(sku_id),
        "owner_id": str(owner_id),
        "owner_type": owner_type.name,
    }
    details = {
        "skuId": str(sku_id),
        "ownerId": str(owner_id),
        "ownerType": owner_type.name,
    }
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await client.create_entitlement(
        discord.Object(id=sku_id), discord.Object(id=owner_id), owner_type
    )
    return json_text({"status": "executed", "action": action, **details})


async def handle_consume_entitlement(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    require_gateway(deps, "consume_entitlement")
    client = _client(deps, "consume_entitlement")

    raw = str(arguments["entitlement_id"])
    entitlement_id = validate_snowflake(raw)
    action = "consume_entitlement"
    # Discord's consume endpoint accepts no audit-log reason, so an optional
    # reason is validated for the gate but never transmitted.
    if arguments.get("reason") is not None:
        require_reason(arguments.get("reason"), action)
    entitlement = await _fetch_entitlement(client, raw, entitlement_id)

    targets = {"entitlement_id": str(entitlement_id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"entitlementId": str(entitlement_id)})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await entitlement.consume()
    return json_text(
        {"status": "executed", "action": action, "entitlementId": str(entitlement_id)}
    )


async def handle_delete_entitlement(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    require_gateway(deps, "delete_entitlement")
    client = _client(deps, "delete_entitlement")

    raw = str(arguments["entitlement_id"])
    entitlement_id = validate_snowflake(raw)
    action = "delete_entitlement"
    # Discord's delete-entitlement endpoint accepts no audit-log reason, so the
    # reason is validated for the gate but never transmitted.
    require_reason(arguments.get("reason"), action)
    entitlement = await _fetch_entitlement(client, raw, entitlement_id)

    targets = {"entitlement_id": str(entitlement_id)}
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(action, targets, {"entitlementId": str(entitlement_id)})
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await entitlement.delete()
    return json_text(
        {"status": "executed", "action": action, "entitlementId": str(entitlement_id)}
    )


async def handle_list_app_commands(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    require_gateway(deps, "list_app_commands")
    tree = _tree(deps, "list_app_commands")

    guild, guild_id = _guild_scope(arguments.get("guild_id"))
    commands = await tree.fetch_commands(guild=guild)
    rows = [_command_row(command) for command in commands]
    return json_text({"guildId": guild_id, "count": len(rows), "commands": rows})


async def handle_get_app_command(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    require_gateway(deps, "get_app_command")
    tree = _tree(deps, "get_app_command")

    raw = str(arguments["command_id"])
    command_id = validate_snowflake(raw)
    guild, _guild_id = _guild_scope(arguments.get("guild_id"))
    try:
        command = await tree.fetch_command(command_id, guild=guild)
    except discord.NotFound:
        raise ValueError(f"Application command '{raw}' not found") from None
    return json_text(_command_row(command))


async def handle_sync_app_commands(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    require_gateway(deps, "sync_app_commands")
    tree = _tree(deps, "sync_app_commands")

    guild, guild_id = _guild_scope(arguments.get("guild_id"))
    action = "sync_app_commands"
    # CommandTree.sync accepts no audit-log reason, so an optional reason is
    # validated for the gate but never transmitted.
    if arguments.get("reason") is not None:
        require_reason(arguments.get("reason"), action)
    targets = {"guild_id": guild_id if guild_id else "global"}
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, {"guildId": guild_id}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    commands = await tree.sync(guild=guild)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "synced": len(commands),
            "guildId": guild_id,
        }
    )
