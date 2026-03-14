import json
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token


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
    return _json(
        {
            "status": "ok",
            "action": "list_auto_moderation_rules",
            "server_id": str(arguments["server_id"]),
            "rules": [],
        }
    )


async def handle_create_auto_moderation_rule(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "applied",
            "action": "create_auto_moderation_rule",
            "server_id": str(arguments["server_id"]),
            "rule": arguments["rule"],
        }
    )


async def handle_update_auto_moderation_rule(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "applied",
            "action": "update_auto_moderation_rule",
            "server_id": str(arguments["server_id"]),
            "rule_id": str(arguments["rule_id"]),
            "rule": arguments["rule"],
        }
    )


async def handle_automod_export_rules(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "status": "ok",
            "action": "automod_export_rules",
            "server_id": str(arguments["server_id"]),
            "export": {"rules": []},
        }
    )
