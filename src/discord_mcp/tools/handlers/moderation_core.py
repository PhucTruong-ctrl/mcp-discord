import json
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token


def _should_require_confirm(arguments: Dict[str, Any]) -> bool:
    return bool(arguments.get("require_confirm", True))


def _is_dry_run(arguments: Dict[str, Any]) -> bool:
    return bool(arguments.get("dry_run", True))


async def handle_moderation_bulk_delete(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    channel_id = str(arguments["channel_id"])
    message_ids = [str(v) for v in arguments["message_ids"]]
    reason = arguments.get("reason")

    targets = {"channel_id": channel_id, "message_ids": sorted(message_ids)}
    action = "moderation_bulk_delete"
    if _is_dry_run(arguments):
        payload = build_dry_run_result(
            action,
            targets,
            {"message_count": len(message_ids), "reason": reason},
        )
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if _should_require_confirm(arguments):
        verify_confirm_token(action, targets, arguments.get("confirm_token"))

    deleted = await gateway.bulk_delete_messages(channel_id, message_ids, reason)
    payload = {"status": "executed", "action": action, "deleted": deleted}
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def handle_moderation_timeout_member(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = str(arguments["server_id"])
    member_id = str(arguments["member_id"])
    duration_minutes = int(arguments["duration_minutes"])
    reason = arguments.get("reason")

    targets = {
        "server_id": server_id,
        "member_id": member_id,
        "duration_minutes": duration_minutes,
    }
    action = "moderation_timeout_member"
    if _is_dry_run(arguments):
        payload = build_dry_run_result(action, targets, {"reason": reason})
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if _should_require_confirm(arguments):
        verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await gateway.timeout_member(server_id, member_id, duration_minutes, reason)
    payload = {"status": "executed", "action": action}
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def handle_moderation_kick_member(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = str(arguments["server_id"])
    member_id = str(arguments["member_id"])
    reason = arguments.get("reason")

    targets = {"server_id": server_id, "member_id": member_id}
    action = "moderation_kick_member"
    if _is_dry_run(arguments):
        payload = build_dry_run_result(action, targets, {"reason": reason})
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if _should_require_confirm(arguments):
        verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await gateway.kick_member(server_id, member_id, reason)
    payload = {"status": "executed", "action": action}
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]


async def handle_moderation_ban_member(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = str(arguments["server_id"])
    member_id = str(arguments["member_id"])
    delete_message_days = int(arguments.get("delete_message_days", 0))
    reason = arguments.get("reason")

    targets = {
        "server_id": server_id,
        "member_id": member_id,
        "delete_message_days": delete_message_days,
    }
    action = "moderation_ban_member"
    if _is_dry_run(arguments):
        payload = build_dry_run_result(action, targets, {"reason": reason})
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if _should_require_confirm(arguments):
        verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await gateway.ban_member(server_id, member_id, delete_message_days, reason)
    payload = {"status": "executed", "action": action}
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
