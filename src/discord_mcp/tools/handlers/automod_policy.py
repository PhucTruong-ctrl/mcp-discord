import json
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.resolve import try_int
from discord_mcp.core.safety import (
    build_dry_run_result,
    verify_confirm_token,
)
from discord_mcp.core.serialize import _serialize_auto_moderation_rule


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
    if "rules" not in ruleset or ruleset["rules"] is None:
        ruleset["rules"] = []
    elif not isinstance(ruleset["rules"], list):
        raise ValueError("ruleset.rules must be an array")


def _build_automod_trigger(rule_data: Dict[str, Any]) -> Any:
    """Build a discord.AutoModTrigger from rule data."""
    import discord

    trigger_type = str(rule_data.get("trigger_type", "keyword")).upper()
    trigger_metadata = rule_data.get("trigger_metadata", {}) or {}

    if trigger_type == "KEYWORD":
        return discord.AutoModTrigger(
            keyword_filter=trigger_metadata.get("keyword_filter", []),
            regex_patterns=trigger_metadata.get("regex_patterns", []),
            allow_list=trigger_metadata.get("allow_list", []),
        )
    if trigger_type == "KEYWORD_PRESET":
        presets_val = trigger_metadata.get("presets", 0)
        return discord.AutoModTrigger(presets=presets_val)
    if trigger_type == "MENTION_SPAM":
        return discord.AutoModTrigger(
            mention_limit=trigger_metadata.get("mention_limit", 10)
        )
    if trigger_type == "MEMBER_PROFILE":
        return discord.AutoModTrigger(
            keyword_filter=trigger_metadata.get("keyword_filter", []),
            regex_patterns=trigger_metadata.get("regex_patterns", []),
            allow_list=trigger_metadata.get("allow_list", []),
        )

    # Default: keyword trigger
    return discord.AutoModTrigger(
        keyword_filter=trigger_metadata.get("keyword_filter", []),
        regex_patterns=trigger_metadata.get("regex_patterns", []),
        allow_list=trigger_metadata.get("allow_list", []),
    )


def _resolve_automod_exemptions(guild: Any, values: Any, attribute: str) -> List[Any]:
    """Resolve exempt role/channel references given as ids or names."""
    import discord

    if not values:
        return []
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{attribute} must be an array of ids or names")

    pool = list(getattr(guild, attribute, []) or [])
    resolved: List[Any] = []
    for value in values:
        key = str(value).strip()
        if not key:
            continue
        target_id = try_int(key)
        if target_id is not None:
            resolved.append(discord.Object(id=target_id))
            continue
        matches = [
            item
            for item in pool
            if str(getattr(item, "name", "")).lower() == key.lower()
        ]
        if len(matches) == 1:
            resolved.append(discord.Object(id=matches[0].id))
            continue
        available = ", ".join(sorted(str(getattr(i, "name", i.id)) for i in pool))
        raise ValueError(
            f"{attribute} entry '{value}' did not match a single "
            f"{'role' if attribute == 'roles' else 'channel'}. Available: {available}"
        )
    return resolved


def _build_automod_actions(
    actions_data: List[Dict[str, Any]],
) -> List[Any]:
    """Build a list of discord.AutoModRuleAction from action data."""
    import discord

    result = []
    for action_data in actions_data:
        action_type = str(action_data.get("type", "block_message")).upper()
        kwargs = {}
        if action_type == "BLOCK_MEMBER_INTERACTION":
            kwargs["type"] = discord.AutoModRuleActionType.block_member_interaction
            custom = action_data.get("custom_message")
            if custom:
                kwargs["custom_message"] = custom
            duration = action_data.get("duration")
            if duration:
                import datetime

                kwargs["duration"] = datetime.timedelta(seconds=int(duration))
        elif action_type == "SEND_ALERT_MESSAGE":
            kwargs["type"] = discord.AutoModRuleActionType.send_alert_message
            channel_id = action_data.get("channel_id")
            if channel_id:
                kwargs["channel_id"] = int(channel_id)
            custom = action_data.get("custom_message")
            if custom:
                kwargs["custom_message"] = custom
        else:
            # BLOCK_MESSAGE
            kwargs["type"] = discord.AutoModRuleActionType.block_message
            custom = action_data.get("custom_message")
            if custom:
                kwargs["custom_message"] = custom
        result.append(discord.AutoModRuleAction(**kwargs))
    return result


def _parse_automod_event_type(event_type_str: str) -> Any:
    """Parse an event type string to discord.AutoModRuleEventType."""
    import discord

    normalized = event_type_str.upper().replace("-", "_")
    try:
        return discord.AutoModRuleEventType[normalized]
    except KeyError:
        return discord.AutoModRuleEventType.message_send


async def handle_automod_validate_ruleset(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    ruleset = arguments["ruleset"]
    _validate_ruleset_shape(ruleset)
    return _json({"status": "valid", "ruleset": ruleset})


async def handle_automod_get_ruleset(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    """Fetch AutoMod rules for a guild from Discord API."""
    gateway = deps.get("gateway")
    if not gateway:
        return _json(
            {
                "guild_id": str(arguments.get("guild_id", "")),
                "ruleset": arguments.get("ruleset", {}),
                "rules": [],
            }
        )
    guild = await gateway.resolve_guild(arguments.get("guild_id"))
    rules = await guild.fetch_automod_rules()
    ruleset_name = str(arguments.get("ruleset_name", "")).strip()
    if ruleset_name:
        rules = [r for r in rules if r.name == ruleset_name]
    serialized = [_serialize_auto_moderation_rule(r) for r in rules]
    return _json({"guild_id": str(guild.id), "rules": serialized})


async def handle_automod_apply_ruleset(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    ruleset = arguments["ruleset"]
    _validate_ruleset_shape(ruleset)
    reason = _required_reason(arguments)
    guild_id = str(arguments["guild_id"])
    ruleset_name = str(ruleset["name"])
    dry_run = bool(arguments.get("dry_run", True))
    action = "automod_apply_ruleset"
    targets = {"guild_id": guild_id, "ruleset_name": ruleset_name, "reason": reason}

    if dry_run:
        payload = build_dry_run_result(
            action,
            targets,
            {"guild_id": guild_id, "ruleset": ruleset, "reason": reason},
        )
        return _json(payload)

    confirm_token = _required_confirm_token(arguments)
    verify_confirm_token(action, targets, confirm_token)

    # Execute: create rules on Discord via gateway (synthetic fallback when unavailable)
    gateway = deps.get("gateway")
    created_rules = []
    errors = []
    if gateway:
        guild = await gateway.resolve_guild(guild_id)
        for rule_data in ruleset.get("rules", []):
            try:
                trigger = _build_automod_trigger(rule_data)
                actions = _build_automod_actions(rule_data.get("actions", []))
                event_type = _parse_automod_event_type(
                    rule_data.get("event_type", "message_send")
                )
                enabled = rule_data.get("enabled", True)
                create_kwargs: Dict[str, Any] = {}
                exempt_roles = _resolve_automod_exemptions(
                    guild, rule_data.get("exempt_roles"), "roles"
                )
                if exempt_roles:
                    create_kwargs["exempt_roles"] = exempt_roles
                exempt_channels = _resolve_automod_exemptions(
                    guild, rule_data.get("exempt_channels"), "channels"
                )
                if exempt_channels:
                    create_kwargs["exempt_channels"] = exempt_channels
                new_rule = await guild.create_automod_rule(
                    name=rule_data["name"],
                    event_type=event_type,
                    trigger=trigger,
                    actions=actions,
                    enabled=enabled,
                    reason=reason,
                    **create_kwargs,
                )
                created_rules.append(_serialize_auto_moderation_rule(new_rule))
            except Exception as exc:
                errors.append(
                    f"Failed to create rule '{rule_data.get('name', '?')}': {exc}"
                )

    status = "applied_with_errors" if errors else "applied"
    response: Dict[str, Any] = {
        "status": status,
        "guild_id": guild_id,
        "ruleset_name": ruleset_name,
        "reason": reason,
        "rules": created_rules,
    }
    if errors:
        response["errors"] = errors
    return _json(response)


async def handle_automod_rollback_ruleset(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    """Rollback AutoMod ruleset.

    Dry-run works as a capability check. The execute path is not supported
    because there is no Discord API primitive for rolling back a ruleset.
    Use automod_get_ruleset to capture a known-good state and
    automod_apply_ruleset to restore it.
    """
    guild_id = str(arguments["guild_id"])
    ruleset_name = str(arguments.get("ruleset_name", "")).strip()
    reason = _required_reason(arguments)
    dry_run = bool(arguments.get("dry_run", True))
    action = "automod_rollback_ruleset"
    targets = {
        "guild_id": guild_id,
        "ruleset_name": ruleset_name,
        "reason": reason,
    }

    if dry_run:
        payload = build_dry_run_result(
            action,
            targets,
            {
                "guild_id": guild_id,
                "ruleset_name": ruleset_name,
                "reason": reason,
            },
        )
        return _json(payload)

    confirm_token = _required_confirm_token(arguments)
    verify_confirm_token(action, targets, confirm_token)

    return _json(
        {
            "status": "not_supported",
            "guild_id": guild_id,
            "ruleset_name": ruleset_name,
            "reason": reason,
            "detail": (
                "AutoMod rollback is not directly supported by the Discord API. "
                "Use automod_get_ruleset to capture a known-good state, then "
                "automod_apply_ruleset to restore it."
            ),
        }
    )
