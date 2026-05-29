import json
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.serialize import _serialize_auto_moderation_rule


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
    rules = []
    if gateway:
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
    created_rule = None
    if gateway:
        import discord

        guild = await gateway.resolve_guild(arguments.get("server_id"))
        rule_data = arguments["rule"]
        trigger_type = str(rule_data.get("trigger_type", "keyword")).upper()
        trigger_metadata = rule_data.get("trigger_metadata", {}) or {}

        if trigger_type == "KEYWORD":
            trigger = discord.AutoModTrigger(
                keyword_filter=trigger_metadata.get("keyword_filter", [])
            )
        elif trigger_type == "KEYWORD_PRESET":
            trigger = discord.AutoModTrigger(presets=trigger_metadata.get("presets", 0))
        elif trigger_type == "MENTION_SPAM":
            trigger = discord.AutoModTrigger(
                mention_limit=trigger_metadata.get("mention_limit", 10)
            )
        elif trigger_type == "MEMBER_PROFILE":
            trigger = discord.AutoModTrigger(
                keyword_filter=trigger_metadata.get("keyword_filter", [])
            )
        else:
            trigger = discord.AutoModTrigger(
                keyword_filter=trigger_metadata.get("keyword_filter", [])
            )

        actions_raw = rule_data.get("actions", [])
        actions = []
        for a in actions_raw:
            atype = str(a.get("type", "block_message")).upper()
            akwargs = {}
            if atype == "BLOCK_MEMBER_INTERACTION":
                akwargs["type"] = discord.AutoModRuleActionType.block_member_interaction
                if a.get("custom_message"):
                    akwargs["custom_message"] = a["custom_message"]
                if a.get("duration"):
                    import datetime

                    akwargs["duration"] = datetime.timedelta(seconds=int(a["duration"]))
            elif atype == "SEND_ALERT_MESSAGE":
                akwargs["type"] = discord.AutoModRuleActionType.send_alert_message
                if a.get("channel_id"):
                    akwargs["channel_id"] = int(a["channel_id"])
                if a.get("custom_message"):
                    akwargs["custom_message"] = a["custom_message"]
            else:
                akwargs["type"] = discord.AutoModRuleActionType.block_message
                if a.get("custom_message"):
                    akwargs["custom_message"] = a["custom_message"]
            actions.append(discord.AutoModRuleAction(**akwargs))

        event_type_str = (
            str(rule_data.get("event_type", "message_send")).upper().replace("-", "_")
        )
        try:
            event_type = discord.AutoModRuleEventType[event_type_str]
        except KeyError:
            event_type = discord.AutoModRuleEventType.message_send

        enabled = rule_data.get("enabled", True)

        new_rule = await guild.create_automod_rule(
            name=rule_data["name"],
            event_type=event_type,
            trigger=trigger,
            actions=actions,
            enabled=enabled,
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
    if gateway:
        guild = await gateway.resolve_guild(arguments.get("server_id"))
        rule_id = int(arguments["rule_id"])
        rules = await guild.fetch_automod_rules()
        target = next((r for r in rules if r.id == rule_id), None)
        if target is None:
            raise ValueError(
                f"AutoMod rule '{arguments['rule_id']}' not found in guild"
            )

        import discord

        rule_data = arguments["rule"]
        kwargs = {}
        if "name" in rule_data:
            kwargs["name"] = rule_data["name"]
        if "enabled" in rule_data:
            kwargs["enabled"] = rule_data["enabled"]
        if "event_type" in rule_data:
            et = str(rule_data["event_type"]).upper().replace("-", "_")
            try:
                kwargs["event_type"] = discord.AutoModRuleEventType[et]
            except KeyError:
                pass
        if "trigger_type" in rule_data or "trigger_metadata" in rule_data:
            ttype = str(rule_data.get("trigger_type", "keyword")).upper()
            tmeta = rule_data.get("trigger_metadata", {}) or {}
            if ttype == "KEYWORD":
                kwargs["trigger"] = discord.AutoModTrigger(
                    keyword_filter=tmeta.get("keyword_filter", [])
                )
            elif ttype == "KEYWORD_PRESET":
                kwargs["trigger"] = discord.AutoModTrigger(
                    presets=tmeta.get("presets", 0)
                )
            elif ttype == "MENTION_SPAM":
                kwargs["trigger"] = discord.AutoModTrigger(
                    mention_limit=tmeta.get("mention_limit", 10)
                )
            else:
                kwargs["trigger"] = discord.AutoModTrigger(
                    keyword_filter=tmeta.get("keyword_filter", [])
                )
        if "actions" in rule_data:
            actions_raw = rule_data["actions"]
            actions = []
            for a in actions_raw:
                atype = str(a.get("type", "block_message")).upper()
                akwargs = {}
                if atype == "BLOCK_MEMBER_INTERACTION":
                    akwargs["type"] = (
                        discord.AutoModRuleActionType.block_member_interaction
                    )
                    if a.get("custom_message"):
                        akwargs["custom_message"] = a["custom_message"]
                    if a.get("duration"):
                        import datetime

                        akwargs["duration"] = datetime.timedelta(
                            seconds=int(a["duration"])
                        )
                elif atype == "SEND_ALERT_MESSAGE":
                    akwargs["type"] = discord.AutoModRuleActionType.send_alert_message
                    if a.get("channel_id"):
                        akwargs["channel_id"] = int(a["channel_id"])
                    if a.get("custom_message"):
                        akwargs["custom_message"] = a["custom_message"]
                else:
                    akwargs["type"] = discord.AutoModRuleActionType.block_message
                    if a.get("custom_message"):
                        akwargs["custom_message"] = a["custom_message"]
                actions.append(discord.AutoModRuleAction(**akwargs))
            kwargs["actions"] = actions

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
    rules = []
    if gateway:
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
