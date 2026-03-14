import json
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.safety import verify_confirm_token


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
        return _json({"status": "dry_run", "action": action, "targets": targets})
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
        return _json({"status": "dry_run", "action": action, "targets": targets})
    verify_confirm_token(action, targets, arguments.get("confirm_token"))
    return _json({"status": "applied", "action": action, "targets": targets})
