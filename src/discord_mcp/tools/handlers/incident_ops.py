import json
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.safety import (
    generate_confirm_token_with_reason,
    validate_confirm_token,
)


def _required_reason(arguments: Dict[str, Any]) -> str:
    reason = str(arguments.get("reason", "")).strip()
    if not reason:
        raise ValueError("reason is required")
    return reason


def _required_confirm_token(arguments: Dict[str, Any]) -> str:
    token = str(arguments.get("confirm_token", "")).strip()
    if not token:
        raise ValueError("confirm_token is required")
    return token


def _json(payload: Dict[str, Any]) -> List[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, sort_keys=True))]


async def handle_incident_get_channel_state(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "channel_id": str(arguments["channel_id"]),
            "state": arguments.get("state", {}),
        }
    )


async def handle_incident_set_channel_state(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    return _json(
        {
            "channel_id": str(arguments["channel_id"]),
            "state": arguments["state"],
        }
    )


async def handle_incident_apply_lockdown(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    reason = _required_reason(arguments)
    channel_ids = [str(channel_id) for channel_id in arguments["channel_ids"]]
    dry_run = bool(arguments.get("dry_run", True))

    if dry_run:
        token = generate_confirm_token_with_reason(
            action="incident_apply_lockdown",
            targets=channel_ids,
            reason=reason,
        )
        return _json(
            {
                "status": "dry_run",
                "action": "incident_apply_lockdown",
                "reason": reason,
                "channel_ids": channel_ids,
                "confirm_token": token,
            }
        )

    confirm_token = _required_confirm_token(arguments)
    validate_confirm_token(
        action="incident_apply_lockdown",
        targets=channel_ids,
        reason=reason,
        confirm_token=confirm_token,
    )
    return _json(
        {
            "status": "applied",
            "reason": reason,
            "channel_ids": channel_ids,
        }
    )


async def handle_incident_rollback_lockdown(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    reason = _required_reason(arguments)
    channel_ids = [str(channel_id) for channel_id in arguments["channel_ids"]]
    dry_run = bool(arguments.get("dry_run", True))

    if dry_run:
        token = generate_confirm_token_with_reason(
            action="incident_rollback_lockdown",
            targets=channel_ids,
            reason=reason,
        )
        return _json(
            {
                "status": "dry_run",
                "action": "incident_rollback_lockdown",
                "reason": reason,
                "channel_ids": channel_ids,
                "confirm_token": token,
            }
        )

    confirm_token = _required_confirm_token(arguments)
    validate_confirm_token(
        action="incident_rollback_lockdown",
        targets=channel_ids,
        reason=reason,
        confirm_token=confirm_token,
    )
    return _json(
        {
            "status": "rolled_back",
            "reason": reason,
            "channel_ids": channel_ids,
        }
    )
