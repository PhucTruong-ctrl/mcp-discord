import json
from typing import Any, Dict, List

import discord
from mcp.types import TextContent

from discord_mcp.core.emoji import parse_emoji, parse_welcome_emoji
from discord_mcp.core.resolve import try_int
from discord_mcp.core.serialize import (
    _serialize_onboarding,
    _serialize_welcome_screen,
)


def _build_welcome_channels(guild: Any, entries: Any) -> List[Any]:
    """Build discord.WelcomeChannel objects from a JSON payload."""
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("welcome_channels must be an array")

    channels = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(
                "each welcome_channels entry needs channel_id and optional description/emoji"
            )
        raw_id = entry.get("channel_id", entry.get("channelId"))
        if raw_id is None:
            raise ValueError(
                "each welcome_channels entry needs channel_id "
                "(optional description, emoji)"
            )
        channel_id = int(raw_id)
        channel = guild.get_channel(channel_id) or discord.Object(id=channel_id)
        # a custom emoji must arrive as an object, or Discord gets a name without an id
        emoji = parse_welcome_emoji(guild, entry)
        channels.append(
            discord.WelcomeChannel(
                channel=channel,
                description=str(entry.get("description") or ""),
                emoji=emoji,
            )
        )
    return channels


async def handle_get_guild_welcome_screen(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(
        arguments.get("server_id") or arguments.get("server")
    )
    try:
        screen = await guild.welcome_screen()
    except discord.NotFound:
        # 10069 Unknown Guild Welcome Screen means "not configured", not a failure.
        screen = None
    payload = {
        "serverId": str(guild.id),
        "serverName": guild.name,
        "configured": screen is not None,
        "welcomeScreen": _serialize_welcome_screen(screen) if screen else None,
        "hint": (
            None
            if screen is not None
            else "No welcome screen configured yet; create one with "
            "update_guild_welcome_screen (the guild must have the COMMUNITY feature)."
        ),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_update_guild_welcome_screen(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(
        arguments.get("server_id") or arguments.get("server")
    )
    ws_args = arguments.get("welcome_screen") or {}
    if not isinstance(ws_args, dict):
        raise ValueError("welcome_screen must be an object")

    # Guild.edit_welcome_screen() works without a pre-existing screen, which
    # WelcomeScreen.edit() cannot do (fetching an unset screen raises NotFound).
    edit_kwargs: Dict[str, Any] = {}
    if "description" in ws_args:
        edit_kwargs["description"] = ws_args["description"]
    if "enabled" in ws_args:
        edit_kwargs["enabled"] = bool(ws_args["enabled"])
    raw_welcome_channels = ws_args.get(
        "welcome_channels", ws_args.get("welcomeChannels")
    )
    if raw_welcome_channels is not None:
        edit_kwargs["welcome_channels"] = _build_welcome_channels(
            guild, raw_welcome_channels
        )
    if arguments.get("reason"):
        edit_kwargs["reason"] = arguments["reason"]

    if not edit_kwargs or set(edit_kwargs) == {"reason"}:
        raise ValueError(
            "nothing to update: pass welcome_screen.description, enabled or welcome_channels"
        )

    updated_screen = await guild.edit_welcome_screen(**edit_kwargs)
    payload = {
        "serverId": str(guild.id),
        "updated": True,
        "configured": True,
        "welcomeScreen": _serialize_welcome_screen(updated_screen),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_get_guild_onboarding(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(
        arguments.get("server_id") or arguments.get("server")
    )
    try:
        onboarding = await guild.onboarding()
    except discord.NotFound:
        raise ValueError(
            f"Server '{guild.id}' has no onboarding configured (unknown guild onboarding)."
        )
    payload = {
        "serverId": str(guild.id),
        "serverName": guild.name,
        "onboarding": _serialize_onboarding(onboarding),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


_PUBLIC_CHANNEL_TYPES = (
    discord.ChannelType.text,
    discord.ChannelType.voice,
    discord.ChannelType.news,
    discord.ChannelType.stage_voice,
    discord.ChannelType.forum,
)


def _onboarding_requirement_report(guild: Any) -> str:
    """Explain Discord's onboarding requirements with the guild's current numbers.

    Discord requires at least 7 public channels and at least 5 of them writable by
    ``@everyone``; a guild that fails this cannot update onboarding at all (error 350001),
    from the API or the client.
    """
    try:
        everyone = guild.default_role
    except Exception:  # noqa: BLE001 - reporting must not fail
        everyone = None
    viewable = []
    writable = []
    for channel in getattr(guild, "channels", []) or []:
        if getattr(channel, "type", None) not in _PUBLIC_CHANNEL_TYPES:
            continue
        try:
            permissions = channel.permissions_for(everyone)
        except Exception:  # noqa: BLE001 - reporting must not fail
            continue
        if permissions.view_channel:
            viewable.append(channel)
            if permissions.send_messages:
                writable.append(channel)
    return (
        f"current state: {len(viewable)} public channels viewable by @everyone "
        f"(need >= 7), {len(writable)} writable by @everyone (need >= 5)"
    )


def _resolve_onboarding_prompt_type(value: Any, where: str) -> Any:
    """Map a caller value onto :class:`discord.OnboardingPromptType`."""
    if isinstance(value, discord.OnboardingPromptType):
        return value
    if value is None:
        return discord.OnboardingPromptType.multiple_choice
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        # accept the dotted form the reader emits, e.g. "OnboardingPromptType.dropdown"
        normalized = normalized.rsplit(".", 1)[-1]
        if normalized in discord.OnboardingPromptType.__members__:
            return discord.OnboardingPromptType[normalized]
        action = try_int(normalized)
        if action is not None:
            return discord.enums.try_enum(discord.OnboardingPromptType, action)
    else:
        action = try_int(value)
        if action is not None:
            return discord.enums.try_enum(discord.OnboardingPromptType, action)
    raise ValueError(
        f"{where}.type must be 'multiple_choice' (0) or 'dropdown' (1), got {value!r}"
    )


def _resolve_onboarding_mode(value: Any) -> Any:
    """Map a caller value onto :class:`discord.OnboardingMode`."""
    if isinstance(value, discord.OnboardingMode):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().rsplit(".", 1)[-1]
        if normalized in discord.OnboardingMode.__members__:
            return discord.OnboardingMode[normalized]
        number = try_int(normalized)
        if number is not None:
            return discord.enums.try_enum(discord.OnboardingMode, number)
    else:
        number = try_int(value)
        if number is not None:
            return discord.enums.try_enum(discord.OnboardingMode, number)
    raise ValueError(f"mode must be 'default' (0) or 'advanced' (1), got {value!r}")


def _first_present(data: Dict[str, Any], *keys: str, default: bool) -> bool:
    for key in keys:
        if key in data and data[key] is not None:
            return bool(data[key])
    return bool(default)


def _channel_id_list(guild: Any, values: Any, where: str) -> List[int]:
    """Resolve channel references (snowflakes or unique names) into ids."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{where} must be an array of ids or names")
    ids = []
    for value in values:
        snowflake = try_int(value)
        if snowflake is not None:
            ids.append(snowflake)
            continue
        wanted = str(value).strip().lower().removeprefix("#")
        matches = [
            channel
            for channel in getattr(guild, "channels", []) or []
            if str(getattr(channel, "name", "")).strip().lower() == wanted
        ]
        if len(matches) == 1:
            ids.append(matches[0].id)
            continue
        if len(matches) > 1:
            raise ValueError(
                f"{where}: channel name '{value}' matches {len(matches)} channels; use the id"
            )
        raise ValueError(f"{where}: channel '{value}' not found in server {guild.id}")
    return ids


def _id_list(values: Any, where: str) -> List[int]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{where} must be an array of ids")
    ids = []
    for value in values:
        snowflake = try_int(value)
        if snowflake is None:
            raise ValueError(f"{where} entries must be snowflake ids, got {value!r}")
        ids.append(snowflake)
    return ids


def _build_onboarding_option(
    data: Any, where: str, guild: Any = None
) -> discord.OnboardingPromptOption:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be an object")
    title = str(data.get("title") or "").strip()
    if not title:
        raise ValueError(f"{where}.title is required")
    token = parse_emoji(guild, data)
    kwargs: Dict[str, Any] = {"title": title}
    if data.get("description") is not None:
        kwargs["description"] = str(data["description"])
    if token:
        kwargs["emoji"] = token
    channel_ids = _id_list(data.get("channel_ids"), f"{where}.channel_ids")
    if channel_ids:
        kwargs["channels"] = channel_ids
    role_ids = _id_list(data.get("role_ids"), f"{where}.role_ids")
    if role_ids:
        kwargs["roles"] = role_ids
    return discord.OnboardingPromptOption(**kwargs)


def _build_onboarding_prompts(
    prompts: Any, guild: Any = None
) -> List[discord.OnboardingPrompt]:
    """Convert the JSON prompt payload into the objects discord.py expects.

    ``Guild.edit_onboarding`` calls ``prompt.to_dict(id=index)`` on each entry, so raw
    dicts raise ``'dict' object has no attribute 'to_dict'``.
    """
    if not isinstance(prompts, list):
        raise ValueError("onboarding.prompts must be an array")
    built = []
    for index, prompt in enumerate(prompts):
        where = f"onboarding.prompts[{index}]"
        if not isinstance(prompt, dict):
            raise ValueError(f"{where} must be an object")
        title = str(prompt.get("title") or "").strip()
        if not title:
            raise ValueError(f"{where}.title is required")
        raw_options = prompt.get("options")
        if not isinstance(raw_options, list) or not raw_options:
            raise ValueError(f"{where}.options must be a non-empty array")
        built.append(
            discord.OnboardingPrompt(
                type=_resolve_onboarding_prompt_type(prompt.get("type"), where),
                title=title,
                options=[
                    _build_onboarding_option(
                        option, f"{where}.options[{option_index}]", guild
                    )
                    for option_index, option in enumerate(raw_options)
                ],
                # accept both the JSON/API spelling and the spelling our reader emits
                single_select=_first_present(
                    prompt, "single_select", "singleSelect", default=True
                ),
                required=bool(prompt.get("required", True)),
                in_onboarding=_first_present(
                    prompt, "in_onboarding", "inOnboarding", default=True
                ),
            )
        )
    return built


async def handle_update_guild_onboarding(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    guild = await deps["gateway"].resolve_guild(
        arguments.get("server_id") or arguments.get("server")
    )
    ob_args = arguments.get("onboarding", {})
    if not isinstance(ob_args, dict):
        raise ValueError("onboarding must be an object")

    known_keys = {
        "enabled",
        "prompts",
        "default_channels",
        "defaultChannels",
        "default_channel_ids",
        "mode",
    }
    if not known_keys.intersection(ob_args):
        raise ValueError(
            "onboarding must contain at least one of: enabled, prompts, default_channels, mode"
        )
    prompts_provided = "prompts" in ob_args
    default_channels_provided = any(
        key in ob_args
        for key in ("default_channels", "defaultChannels", "default_channel_ids")
    )

    # The endpoint is a PUT: a field left out of the request is replaced with "empty" and
    # Discord then rejects the write with 350001 (Cannot update onboarding while below
    # requirements). Merge the caller's changes onto the current configuration - read from
    # the raw payload so channel order, prompt ids and animated emoji detail survive - and
    # always send every field.
    try:
        current = await deps["gateway"].fetch_onboarding_payload(
            arguments.get("server_id") or arguments.get("server")
        )
    except discord.NotFound:
        raise ValueError(
            f"Server '{guild.id}' has no onboarding configured (unknown guild onboarding)."
        )

    prompts = (
        _build_onboarding_prompts(ob_args["prompts"], guild)
        if prompts_provided
        else _build_onboarding_prompts(current.get("prompts") or [], guild)
    )
    if default_channels_provided:
        raw_default_channels = next(
            ob_args[key]
            for key in (
                "default_channels",
                "defaultChannels",
                "default_channel_ids",
                "defaultChannelIds",
            )
            if key in ob_args
        )
    else:
        raw_default_channels = (
            current.get("default_channel_ids") or current.get("defaultChannelIds") or []
        )
    # discord.py reads .id off each entry; names are accepted too so a verbatim read
    # (which also carries display names) round trips
    default_channels = [
        discord.Object(id=channel_id)
        for channel_id in _channel_id_list(
            guild, raw_default_channels, "onboarding.default_channels"
        )
    ]

    try:
        updated_onboarding = await guild.edit_onboarding(
            prompts=prompts,
            default_channels=default_channels,
            enabled=(
                bool(ob_args["enabled"])
                if "enabled" in ob_args
                else bool(current.get("enabled", False))
            ),
            mode=(
                _resolve_onboarding_mode(ob_args["mode"])
                if ob_args.get("mode") is not None
                else _resolve_onboarding_mode(current.get("mode", 0))
            ),
            reason=arguments.get("reason"),
        )
    except discord.HTTPException as exc:
        if getattr(exc, "code", None) in (350000, 350001):
            raise ValueError(
                "Discord refuses to update onboarding for this server: it does not meet the "
                "onboarding requirements (>= 7 public channels, >= 5 of them writable by "
                f"@everyone). {_onboarding_requirement_report(guild)}. Make enough channels "
                "writable for @everyone (or temporarily open them), then retry; the Discord "
                "client is blocked by the same rule."
            )
        raise
    payload = {
        "serverId": str(guild.id),
        "updated": True,
        "onboarding": _serialize_onboarding(updated_onboarding),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_dynamic_role_provision(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments.get("server_id") or arguments.get("server")
    user_id = str(arguments["user_id"])
    ruleset = arguments.get("ruleset") or []
    facts = arguments.get("facts") or {}
    reason = arguments.get("reason")

    member = await gateway.resolve_member(user_id, server_id)
    applied_role_ids: List[str] = []
    skipped: List[Dict[str, str]] = []

    for rule in ruleset:
        condition_key = str(rule["condition"])
        role_id = str(rule["role_id"])
        op = str(rule["op"])
        if not bool(facts.get(condition_key)):
            skipped.append({"role_id": role_id, "reason": "condition_not_met"})
            continue
        role = await gateway.resolve_role(role_id, server_id)
        if op == "add":
            await member.add_roles(role, reason=reason)
            applied_role_ids.append(role_id)
        elif op == "remove":
            await member.remove_roles(role, reason=reason)
            applied_role_ids.append(role_id)
        else:
            skipped.append({"role_id": role_id, "reason": f"unsupported_op:{op}"})

    payload = {
        "appliedRoleIds": applied_role_ids,
        "skipped": skipped,
        "reason": reason,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


def _evaluate_gate(gate: Dict[str, Any], facts: Dict[str, Any]) -> bool | None:
    gate_type = gate.get("type")
    config = gate.get("config") or {}
    if gate_type == "membership_age":
        actual_days = int(facts.get("membership_age_days", 0))
        required_days = int(config.get("min_days", 0))
        return actual_days >= required_days
    if gate_type == "has_role":
        role_id = str(config.get("role_id", ""))
        role_ids = {str(role) for role in facts.get("role_ids", [])}
        return role_id in role_ids
    if gate_type == "manual_approve":
        return None
    raise ValueError(f"Unsupported gate type: {gate_type}")


async def handle_verification_gate_orchestrator(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gates = arguments.get("gates") or []
    mode = arguments.get("mode", "all")
    facts = arguments.get("facts") or {}

    passed_gates: List[str] = []
    failed_gates: List[str] = []
    pending_gates: List[str] = []

    for index, gate in enumerate(gates):
        gate_key = f"{gate.get('type')}:{index}"
        outcome = _evaluate_gate(gate, facts)
        if outcome is True:
            passed_gates.append(gate_key)
        elif outcome is False:
            failed_gates.append(gate_key)
        else:
            pending_gates.append(gate_key)

    if pending_gates:
        status = "pending"
    elif mode == "all":
        status = "passed" if not failed_gates else "failed"
    else:
        status = "passed" if passed_gates else "failed"

    next_action = "manual_review" if status == "pending" else "none"
    payload = {
        "status": status,
        "passedGates": passed_gates,
        "failedGates": failed_gates,
        "nextAction": next_action,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_progressive_access_unlock(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    policy = arguments["policy"]
    facts = arguments.get("facts") or {}
    completed = {str(x) for x in facts.get("requirements_completed", [])}

    unlocked: List[Dict[str, str]] = []
    for unlock in policy.get("unlocks", []):
        required = {str(x) for x in unlock.get("requires", [])}
        if required.issubset(completed):
            unlocked.append({"type": str(unlock["type"]), "id": str(unlock["id"])})

    requirements = [str(x) for x in policy.get("requirements", [])]
    remaining = [
        requirement for requirement in requirements if requirement not in completed
    ]
    payload = {
        "unlocked": unlocked,
        "remainingRequirements": remaining,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_onboarding_friction_audit(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    server_id = arguments.get("server_id")
    window_days = arguments.get("window_days")
    stage_stats = arguments.get("stage_stats") or []
    stages = []
    for stage in stage_stats:
        entered = int(stage.get("entered", 0))
        completed = int(stage.get("completed", 0))
        drop_rate = 0.0 if entered <= 0 else (entered - completed) / entered
        stages.append(
            {
                "stage": stage.get("stage"),
                "entered": entered,
                "completed": completed,
                "dropRate": round(drop_rate, 4),
            }
        )

    total_entered = sum(stage["entered"] for stage in stages)
    total_completed = sum(stage["completed"] for stage in stages)
    completion_rate = 0.0 if total_entered <= 0 else total_completed / total_entered
    payload = {
        "serverId": str(server_id) if server_id is not None else None,
        "windowDays": int(window_days) if window_days is not None else None,
        "dropOffStages": stages,
        "completionRate": round(completion_rate, 4),
        "recommendations": [
            "Focus on stages with highest dropRate",
            "Reduce mandatory steps in early onboarding",
        ],
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]
