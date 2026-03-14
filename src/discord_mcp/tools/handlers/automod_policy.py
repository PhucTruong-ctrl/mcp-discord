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


def _validate_ruleset_shape(ruleset: Dict[str, Any]) -> None:
    if not isinstance(ruleset, dict):
        raise ValueError("ruleset must be an object")
    if not str(ruleset.get("name", "")).strip():
        raise ValueError("ruleset.name is required")
    rules = ruleset.get("rules")
    if not isinstance(rules, list):
        raise ValueError("ruleset.rules must be an array")


async def handle_automod_validate_ruleset(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    ruleset = arguments["ruleset"]
    _validate_ruleset_shape(ruleset)
    return _json({"status": "valid", "ruleset": ruleset})


async def handle_automod_get_ruleset(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    ruleset = arguments["ruleset"]
    _validate_ruleset_shape(ruleset)
    return _json({"guild_id": str(arguments["guild_id"]), "ruleset": ruleset})


async def handle_automod_apply_ruleset(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    ruleset = arguments["ruleset"]
    _validate_ruleset_shape(ruleset)
    reason = _required_reason(arguments)
    guild_id = str(arguments["guild_id"])
    ruleset_name = str(ruleset["name"])
    dry_run = bool(arguments.get("dry_run", True))

    if dry_run:
        token = generate_confirm_token_with_reason(
            action="automod_apply_ruleset",
            targets=[guild_id, ruleset_name],
            reason=reason,
        )
        return _json(
            {
                "status": "dry_run",
                "guild_id": guild_id,
                "ruleset": ruleset,
                "reason": reason,
                "confirm_token": token,
            }
        )

    confirm_token = _required_confirm_token(arguments)
    validate_confirm_token(
        action="automod_apply_ruleset",
        targets=[guild_id, ruleset_name],
        reason=reason,
        confirm_token=confirm_token,
    )
    return _json(
        {
            "status": "applied",
            "guild_id": guild_id,
            "ruleset_name": ruleset_name,
            "reason": reason,
        }
    )


async def handle_automod_rollback_ruleset(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    reason = _required_reason(arguments)
    guild_id = str(arguments["guild_id"])
    ruleset_name = str(arguments["ruleset_name"])
    dry_run = bool(arguments.get("dry_run", True))

    if dry_run:
        token = generate_confirm_token_with_reason(
            action="automod_rollback_ruleset",
            targets=[guild_id, ruleset_name],
            reason=reason,
        )
        return _json(
            {
                "status": "dry_run",
                "guild_id": guild_id,
                "ruleset_name": ruleset_name,
                "reason": reason,
                "confirm_token": token,
            }
        )

    confirm_token = _required_confirm_token(arguments)
    validate_confirm_token(
        action="automod_rollback_ruleset",
        targets=[guild_id, ruleset_name],
        reason=reason,
        confirm_token=confirm_token,
    )
    return _json(
        {
            "status": "rolled_back",
            "guild_id": guild_id,
            "ruleset_name": ruleset_name,
            "reason": reason,
        }
    )
