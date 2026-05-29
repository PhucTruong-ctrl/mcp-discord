import json
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.serialize import _serialize_auto_moderation_rule
from discord_mcp.tools.handlers.automod_policy import (
    _build_automod_actions,
    _build_automod_trigger,
    _parse_automod_event_type,
)


def _json(payload: Dict[str, Any]) -> List[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def handle_bulk_ban_members(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = str(arguments["server_id"])
    member_ids = sorted(str(member_id) for member_id in arguments["member_ids"])
    action = "bulk_ban_members"
    targets = {"server_id": server_id, "member_ids": member_ids}
    if bool(arguments.get("dry_run", True)):
        return _json(build_dry_run_result(action, targets, {"reason": ""}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))
    return _json({"status": "applied", "action": action, "targets": targets})


async def handle_prune_inactive_members(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = str(arguments["server_id"])
    days = int(arguments["days"])
    action = "prune_inactive_members"
    targets = {"server_id": server_id, "days": days}
    if bool(arguments.get("dry_run", True)):
        return _json(build_dry_run_result(action, targets, {"reason": ""}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))
    return _json({"status": "applied", "action": action, "targets": targets})


async def handle_remove_member_timeout(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "applied",
            "action": "remove_member_timeout",
            "targets": {
                "server_id": str(arguments["server_id"]),
                "member_id": str(arguments["member_id"]),
            },
        }
    )


async def handle_unban_member(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "applied",
            "action": "unban_member",
            "targets": {
                "server_id": str(arguments["server_id"]),
                "member_id": str(arguments["member_id"]),
            },
            "reason": str(arguments.get("reason", "")).strip(),
        }
    )


async def handle_create_category(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "applied",
            "action": "create_category",
            "server_id": str(arguments["server_id"]),
            "name": str(arguments["name"]),
        }
    )


async def handle_rename_category(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "applied",
            "action": "rename_category",
            "category_id": str(arguments["category_id"]),
            "name": str(arguments["name"]),
        }
    )


async def handle_move_category(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "applied",
            "action": "move_category",
            "category_id": str(arguments["category_id"]),
            "position": int(arguments["position"]),
        }
    )


async def handle_delete_category(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    category_id = str(arguments["category_id"])
    action = "delete_category"
    targets = {"category_id": category_id}
    if bool(arguments.get("dry_run", True)):
        return _json(build_dry_run_result(action, targets, {"reason": ""}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))
    return _json({"status": "applied", "action": action, "targets": targets})


async def handle_create_incident_room(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "applied",
            "action": "create_incident_room",
            "server_id": str(arguments["server_id"]),
            "name": str(arguments["name"]),
            "reason": str(arguments["reason"]),
        }
    )


async def handle_append_incident_event(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "applied",
            "action": "append_incident_event",
            "incident_channel_id": str(arguments["incident_channel_id"]),
            "event_text": str(arguments["event_text"]),
            "severity": str(arguments["severity"]),
        }
    )


async def handle_close_incident(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "applied",
            "action": "close_incident",
            "incident_channel_id": str(arguments["incident_channel_id"]),
            "summary": str(arguments["summary"]),
            "reason": str(arguments["reason"]),
        }
    )


async def handle_list_auto_moderation_rules(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    if not gateway:
        return _json(
            {
                "status": "ok",
                "server_id": str(arguments.get("server_id", "")),
                "rules": [],
            }
        )
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
        return _json(
            {
                "status": "applied",
                "action": "create_auto_moderation_rule",
                "server_id": str(arguments.get("server_id", "")),
                "rule": {"name": arguments.get("rule", {}).get("name", "")},
            }
        )
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
        return _json(
            {
                "status": "applied",
                "action": "update_auto_moderation_rule",
                "server_id": str(arguments.get("server_id", "")),
                "rule_id": str(arguments.get("rule_id", "")),
            }
        )
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
        return _json(
            {
                "status": "ok",
                "server_id": str(arguments.get("server_id", "")),
                "export": {"rules": []},
            }
        )
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
