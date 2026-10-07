import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import discord
from mcp.types import TextContent

from discord_mcp.core.permissions import overwrite_index
from discord_mcp.core.resolve import try_int
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.state import get_channel_state, set_channel_state
from discord_mcp.core.serialize import _serialize_auto_moderation_rule
from discord_mcp.tools.handlers.automod_policy import (
    _automod_exempt_kwargs,
    _build_automod_actions,
    _build_automod_trigger,
    _parse_automod_event_type,
)


def _json(payload: Dict[str, Any]) -> List[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def _resolve_member(gateway: Any, server_id: str, member_id: str):
    guild = await gateway.resolve_guild(server_id)
    try:
        return await guild.fetch_member(int(member_id))
    except Exception as exc:  # noqa: BLE001 - surfaced with the member id
        raise ValueError(f"Member '{member_id}' not found in server {server_id}: {exc}")


async def _resolve_channel(gateway: Any, channel_id: str, expect: str = "channel"):
    channel = await gateway.fetch_channel(str(channel_id))
    if channel is None:
        raise ValueError(f"{expect.capitalize()} '{channel_id}' not found")
    return channel


def _is_category_type(channel: Any) -> bool:
    return "category" in str(getattr(channel, "type", ""))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _channel_everyone_overwrite(channel: Any) -> Dict[str, Any]:
    """Snapshot of the @everyone overwrite so a later change can be reverted."""
    from discord_mcp.core.permissions import as_permission_bits

    index = overwrite_index(channel)
    everyone_id = str(
        getattr(
            getattr(getattr(channel, "guild", None), "default_role", None), "id", ""
        )
    )
    overwrite = index.get(everyone_id)
    if overwrite is None:
        return {"existed": False, "allow": 0, "deny": 0}
    allow, deny = overwrite.pair()
    return {
        "existed": True,
        "allow": as_permission_bits(allow),
        "deny": as_permission_bits(deny),
    }


async def handle_bulk_ban_members(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    server_id = str(arguments["server_id"])
    member_ids = sorted(str(member_id) for member_id in arguments["member_ids"])
    reason = str(arguments.get("reason", "")).strip() or None
    delete_message_days = int(arguments.get("delete_message_days", 0) or 0)
    action = "bulk_ban_members"
    targets = {
        "server_id": server_id,
        "member_ids": member_ids,
        "delete_message_days": delete_message_days,
    }

    if bool(arguments.get("dry_run", True)):
        return _json(
            build_dry_run_result(
                action,
                targets,
                {
                    "reason": reason or "",
                    "member_count": len(member_ids),
                    "delete_message_days": delete_message_days,
                },
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))
    if not gateway:
        raise ValueError("gateway is required to ban members")
    if not member_ids:
        raise ValueError("member_ids must be a non-empty array")

    banned = await gateway.bulk_ban_members(
        server_id, member_ids, reason, delete_message_days
    )
    return _json(
        {
            "status": "executed",
            "action": action,
            "serverId": server_id,
            "bannedCount": banned,
            "memberIds": member_ids,
        }
    )


async def handle_prune_inactive_members(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    server_id = str(arguments["server_id"])
    days = int(arguments["days"])
    reason = str(arguments.get("reason", "")).strip() or None
    action = "prune_inactive_members"
    targets = {"server_id": server_id, "days": days}

    if bool(arguments.get("dry_run", True)):
        return _json(
            build_dry_run_result(
                action, targets, {"reason": reason or "", "days": days}
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))
    if not gateway:
        raise ValueError("gateway is required to prune members")

    pruned = await gateway.prune_inactive_members(server_id, days, reason)
    return _json(
        {
            "status": "executed",
            "action": action,
            "serverId": server_id,
            "days": days,
            "prunedCount": pruned,
        }
    )


async def handle_remove_member_timeout(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to remove a timeout")
    server_id = str(arguments["server_id"])
    member_id = str(arguments["member_id"])
    reason = str(arguments.get("reason", "")).strip() or None

    member = await _resolve_member(gateway, server_id, member_id)
    await gateway.timeout_member(server_id, member_id, 0, reason)
    return _json(
        {
            "status": "executed",
            "action": "remove_member_timeout",
            "serverId": server_id,
            "memberId": member_id,
            "member": str(member),
            "timedOutUntil": None,
        }
    )


async def handle_unban_member(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to unban a member")
    server_id = str(arguments["server_id"])
    member_id = str(arguments["member_id"])
    reason = str(arguments.get("reason", "")).strip() or None

    guild = await gateway.resolve_guild(server_id)
    try:
        ban = await guild.fetch_ban(discord.Object(id=int(member_id)))
    except Exception as exc:  # noqa: BLE001 - not banned is a user error, not a crash
        raise ValueError(
            f"User '{member_id}' is not banned in server {server_id}: {exc}"
        )
    user = str(getattr(ban, "user", member_id))
    await gateway.unban_member(server_id, member_id, reason)
    return _json(
        {
            "status": "executed",
            "action": "unban_member",
            "serverId": server_id,
            "memberId": member_id,
            "user": user,
            "reason": reason or "",
        }
    )


async def handle_create_category(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to create a category")
    server_id = str(arguments["server_id"])
    name = str(arguments["name"])
    reason = str(arguments.get("reason", "")).strip() or None
    position = arguments.get("position")

    guild = await gateway.resolve_guild(server_id)
    kwargs: Dict[str, Any] = {"name": name, "reason": reason}
    if try_int(position) is not None:
        kwargs["position"] = int(position)
    category = await guild.create_category(**kwargs)
    return _json(
        {
            "status": "executed",
            "action": "create_category",
            "serverId": server_id,
            "categoryId": str(category.id),
            "name": category.name,
            "position": getattr(category, "position", None),
        }
    )


async def handle_rename_category(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to rename a category")
    category = await _resolve_channel(
        gateway, str(arguments["category_id"]), expect="category"
    )
    if not _is_category_type(category):
        raise ValueError(f"Channel '{arguments['category_id']}' is not a category")
    previous = category.name
    reason = str(arguments.get("reason", "")).strip() or None
    await category.edit(name=str(arguments["name"]), reason=reason)
    return _json(
        {
            "status": "executed",
            "action": "rename_category",
            "categoryId": str(category.id),
            "previousName": previous,
            "name": str(arguments["name"]),
        }
    )


async def handle_move_category(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to move a category")
    category = await _resolve_channel(
        gateway, str(arguments["category_id"]), expect="category"
    )
    if not _is_category_type(category):
        raise ValueError(f"Channel '{arguments['category_id']}' is not a category")
    previous = getattr(category, "position", None)
    reason = str(arguments.get("reason", "")).strip() or None
    position = int(arguments["position"])
    await category.edit(position=position, reason=reason)
    return _json(
        {
            "status": "executed",
            "action": "move_category",
            "categoryId": str(category.id),
            "previousPosition": previous,
            "position": position,
        }
    )


async def handle_delete_category(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    category_id = str(arguments["category_id"])
    reason = str(arguments.get("reason", "")).strip() or None
    action = "delete_category"
    targets = {"category_id": category_id}

    if bool(arguments.get("dry_run", True)):
        return _json(build_dry_run_result(action, targets, {"reason": reason or ""}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))
    if not gateway:
        raise ValueError("gateway is required to delete a category")

    category = await _resolve_channel(gateway, category_id, expect="category")
    if not _is_category_type(category):
        raise ValueError(f"Channel '{category_id}' is not a category")
    children = [
        str(channel.id)
        for channel in getattr(category.guild, "channels", [])
        if str(getattr(channel, "category_id", "")) == category_id
    ]
    await category.delete(reason=reason)
    return _json(
        {
            "status": "executed",
            "action": action,
            "categoryId": category_id,
            "orphanedChannelIds": children,
        }
    )


async def handle_create_incident_room(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to create an incident room")
    server_id = str(arguments["server_id"])
    name = str(arguments["name"])
    reason = str(arguments["reason"]).strip()
    category_id = arguments.get("category_id")

    guild = await gateway.resolve_guild(server_id)
    category = None
    if try_int(category_id) is not None:
        category = guild.get_channel(int(category_id))
        if category is None or not _is_category_type(category):
            raise ValueError(
                f"Category '{category_id}' not found in server {server_id}"
            )

    channel = await guild.create_text_channel(
        name=name,
        category=category,
        topic=f"Incident room: {reason}",
        reason=reason,
    )
    state = set_channel_state(
        str(channel.id),
        {
            "kind": "incident_room",
            "reason": reason,
            "createdFrom": "create_incident_room",
            "events": [],
        },
    )
    await channel.send(
        f"**Incident room opened** - {reason}\n"
        f"Use `append_incident_event` to log events and `close_incident` to close."
    )
    return _json(
        {
            "status": "executed",
            "action": "create_incident_room",
            "serverId": server_id,
            "channelId": str(channel.id),
            "name": channel.name,
            "reason": reason,
            "state": state,
        }
    )


async def handle_append_incident_event(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to log an incident event")
    channel_id = str(arguments["incident_channel_id"])
    event_text = str(arguments["event_text"])
    severity = str(arguments["severity"]).lower()

    channel = await _resolve_channel(gateway, channel_id, expect="channel")
    marker = {"low": "-", "medium": "!", "high": "!!", "critical": "!!!"}.get(
        severity, severity
    )
    message = await channel.send(f"**[{severity.upper()}]** {marker} {event_text}")

    state = get_channel_state(channel_id)
    events = list(state.get("events", []))
    events.append(
        {
            "text": event_text,
            "severity": severity,
            "messageId": str(message.id),
            "at": message.created_at.isoformat(),
        }
    )
    state.update({"kind": state.get("kind", "incident_room"), "events": events})
    set_channel_state(channel_id, state)
    return _json(
        {
            "status": "executed",
            "action": "append_incident_event",
            "channelId": channel_id,
            "messageId": str(message.id),
            "severity": severity,
            "eventCount": len(events),
        }
    )


async def handle_close_incident(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to close an incident")
    channel_id = str(arguments["incident_channel_id"])
    summary = str(arguments["summary"])
    reason = str(arguments["reason"]).strip()

    channel = await _resolve_channel(gateway, channel_id, expect="channel")
    state = get_channel_state(channel_id)
    previous_overwrite = _channel_everyone_overwrite(channel)

    await channel.send(
        f"**Incident closed** - {reason}\n\n**Summary**\n{summary}\n\n"
        f"Events logged: {len(state.get('events', []))}"
    )
    # keep history readable but stop further chatter
    await channel.set_permissions(
        channel.guild.default_role, send_messages=False, reason=reason
    )
    state.update(
        {
            "kind": state.get("kind", "incident_room"),
            "closed": True,
            "closedAt": _now(),
            "closeReason": reason,
            "summary": summary,
            "previousEveryoneOverwrite": previous_overwrite,
        }
    )
    set_channel_state(channel_id, state)
    return _json(
        {
            "status": "executed",
            "action": "close_incident",
            "channelId": channel_id,
            "summary": summary,
            "reason": reason,
            "eventCount": len(state.get("events", [])),
            "sendMessagesDeniedFor": str(channel.guild.default_role.id),
        }
    )


async def handle_list_auto_moderation_rules(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to list AutoMod rules")
    guild = await gateway.resolve_guild(arguments.get("server_id"))
    raw_rules = await guild.fetch_automod_rules()
    rules = [_serialize_auto_moderation_rule(r) for r in raw_rules]
    return _json(
        {
            "status": "ok",
            "server_id": str(arguments.get("server_id", "")),
            "rules": rules,
        }
    )


async def handle_create_auto_moderation_rule(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to create an AutoMod rule")
    guild = await gateway.resolve_guild(arguments.get("server_id"))
    rule_data = arguments["rule"]
    reason = str(arguments.get("reason", "")).strip() or None

    trigger = _build_automod_trigger(rule_data)
    actions = _build_automod_actions(rule_data.get("actions", []))
    event_type = _parse_automod_event_type(rule_data.get("event_type", "message_send"))
    enabled = rule_data.get("enabled", True)

    new_rule = await guild.create_automod_rule(
        name=rule_data["name"],
        event_type=event_type,
        trigger=trigger,
        actions=actions,
        enabled=enabled,
        reason=reason,
        **_automod_exempt_kwargs(guild, rule_data),
    )
    created_rule = _serialize_auto_moderation_rule(new_rule)

    return _json(
        {
            "status": "applied",
            "server_id": str(arguments.get("server_id", "")),
            "rule": created_rule,
        }
    )


async def handle_update_auto_moderation_rule(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to update an AutoMod rule")
    guild = await gateway.resolve_guild(arguments.get("server_id"))
    rule_id = int(arguments["rule_id"])
    rules = await guild.fetch_automod_rules()
    target = next((r for r in rules if r.id == rule_id), None)
    if target is None:
        raise ValueError(f"AutoMod rule '{arguments['rule_id']}' not found in guild")

    reason = str(arguments.get("reason", "")).strip() or None
    rule_data = arguments["rule"]
    kwargs: Dict[str, Any] = {}
    if "name" in rule_data:
        kwargs["name"] = rule_data["name"]
    if "enabled" in rule_data:
        kwargs["enabled"] = rule_data["enabled"]
    if "event_type" in rule_data:
        kwargs["event_type"] = _parse_automod_event_type(rule_data["event_type"])
    if "trigger_type" in rule_data or "trigger_metadata" in rule_data:
        kwargs["trigger"] = _build_automod_trigger(rule_data)
    if "actions" in rule_data:
        kwargs["actions"] = _build_automod_actions(rule_data["actions"])
    kwargs.update(_automod_exempt_kwargs(guild, rule_data))
    if reason:
        kwargs["reason"] = reason

    await target.edit(**kwargs)

    return _json(
        {
            "status": "applied",
            "server_id": str(arguments.get("server_id", "")),
            "rule_id": str(arguments.get("rule_id", "")),
        }
    )


async def handle_automod_export_rules(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        raise ValueError("gateway is required to export AutoMod rules")
    guild = await gateway.resolve_guild(arguments.get("server_id"))
    raw_rules = await guild.fetch_automod_rules()
    rules = [_serialize_auto_moderation_rule(r) for r in raw_rules]
    return _json(
        {
            "status": "ok",
            "server_id": str(arguments.get("server_id", "")),
            "export": {"rules": rules},
        }
    )
