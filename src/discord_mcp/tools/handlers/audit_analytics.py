import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List
import discord

from mcp.types import TextContent

from discord_mcp.core.permissions import as_permission_bits, permission_names
from discord_mcp.core.validation import validate_limit
from discord_mcp.core.resolve import try_int
from discord_mcp.services.discord_gateway import resolve_audit_action

_PERMISSION_FIELDS = ("permissions", "allow", "deny")


def _display_name(obj: Any) -> Any:
    for attribute in ("display_name", "nick", "name", "username"):
        value = getattr(obj, attribute, None)
        if value:
            return str(value)
    return None


def _json_safe(value: Any) -> Any:
    """Best-effort JSON conversion of discord.py audit-log diff values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if getattr(value, "id", None) is not None:
        return {
            "id": str(value.id),
            "name": _display_name(value),
            "type": type(value).__name__,
        }
    inner = getattr(value, "value", None)
    if inner is not None and not callable(inner):
        return _json_safe(inner)
    fields = getattr(value, "__dict__", None)
    if fields:
        return {str(key): _json_safe(item) for key, item in fields.items()}
    return str(value)


def _target_id(entry: Any) -> str:
    """Raw audit target snowflake, safe for entries whose ``target_id`` is null."""
    target_id = getattr(entry, "_target_id", None)
    return str(target_id) if target_id is not None else ""


def _target_snapshot(entry: Any) -> Dict[str, Any]:
    """Target metadata without touching discord.py's fragile target conversion.

    Entries such as member_move carry ``target_id: null``; resolving ``entry.target``
    then raises TypeError inside discord.py. The raw snowflake, the optional
    ``extra`` payload and ``changes`` are always safe to read.
    """
    target_id = _target_id(entry)
    target = None
    if target_id:
        try:
            target = entry.target
        except Exception:  # noqa: BLE001 - discord.py conversion must not break the tool
            target = None
    return {
        "targetId": target_id or None,
        "targetName": _display_name(target) if target is not None else None,
        "targetType": type(target).__name__ if target is not None else None,
        "extra": _json_safe(getattr(entry, "extra", None)),
    }


def _changes_payload(entry: Any) -> Dict[str, Any]:
    """before/after diff for one entry plus decoded permission-bit changes."""
    try:
        changes = entry.changes
        raw_before = dict(changes.before)
        raw_after = dict(changes.after)
    except Exception as exc:  # noqa: BLE001 - one malformed entry must not fail the log
        return {"before": {}, "after": {}, "permissionChanges": [], "error": str(exc)}

    return {
        "before": {str(key): _json_safe(value) for key, value in raw_before.items()},
        "after": {str(key): _json_safe(value) for key, value in raw_after.items()},
        "permissionChanges": _permission_changes(raw_before, raw_after),
    }


def _permission_changes(
    before: Dict[str, Any], after: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Decoded permission-bit deltas, e.g. a role gaining mention_everyone."""
    deltas = []
    for field in _PERMISSION_FIELDS:
        if field not in before and field not in after:
            continue
        before_bits = as_permission_bits(before.get(field))
        after_bits = as_permission_bits(after.get(field))
        if before_bits == after_bits:
            continue
        before_names = permission_names(before_bits)
        after_names = permission_names(after_bits)
        deltas.append(
            {
                "field": field,
                "before": before_bits,
                "after": after_bits,
                "added": [name for name in after_names if name not in before_names],
                "removed": [name for name in before_names if name not in after_names],
            }
        )
    return deltas


def _serialize_entry(entry: Any) -> Dict[str, Any]:
    action = entry.action
    entry_payload = {
        "action": getattr(action, "name", str(action)),
        "actionId": getattr(action, "value", None),
        "actorId": str(entry.user.id) if getattr(entry, "user", None) else None,
        "reason": entry.reason,
        "timestamp": entry.created_at.isoformat(),
    }
    entry_payload.update(_target_snapshot(entry))
    entry_payload["changes"] = _changes_payload(entry)
    return entry_payload


def _within_window(entries: Iterable[Any], window_hours: int) -> List[Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    return [entry for entry in entries if entry.created_at >= cutoff]


def _snowflake(value: Any, field: str) -> int:
    parsed = try_int(value)
    if parsed is None:
        raise ValueError(f"{field} must be a snowflake id, got {value!r}")
    return parsed


async def _fetch_audit_entries(guild: Any, **kwargs: Any) -> List[Any]:
    return [entry async for entry in guild.audit_logs(**kwargs)]


def _audit_kwargs(arguments: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """Pagination kwargs for ``Guild.audit_logs``, omitted when not provided."""
    kwargs: Dict[str, Any] = {"limit": limit}
    for field in ("before", "after"):
        if arguments.get(field) is not None:
            kwargs[field] = discord.Object(id=_snowflake(arguments[field], field))
    if arguments.get("oldest_first") is not None:
        kwargs["oldest_first"] = bool(arguments["oldest_first"])
    return kwargs




async def handle_get_audit_log(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    limit = validate_limit(arguments.get("limit"), 50, 1000)
    action_type = arguments.get("action_type")
    kwargs = _audit_kwargs(arguments, limit)
    if action_type:
        kwargs["action"] = resolve_audit_action(action_type)
    entries = await _fetch_audit_entries(guild, **kwargs)

    payload = {
        "serverId": str(arguments["server_id"]),
        "actionType": action_type or None,
        "entryCount": len(entries),
        "entries": [_serialize_entry(entry) for entry in entries],
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]




async def handle_get_member_moderation_history(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    user_id = str(arguments["user_id"])
    limit = validate_limit(arguments.get("limit"), 200, 1000)
    # Discord filters server-side on ``user``; a client-side filter would only see
    # the newest page and silently truncate older events.
    entries = await _fetch_audit_entries(
        guild,
        limit=limit,
        user=discord.Object(id=_snowflake(user_id, "user_id")),
    )

    payload = {
        "serverId": str(arguments["server_id"]),
        "targetUserId": user_id,
        "eventCount": len(entries),
        "events": [_serialize_entry(entry) for entry in entries],
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_get_channel_activity_summary(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments["server_id"]
    channel_id = str(arguments["channel_id"])
    window_hours = int(arguments.get("window_hours", 24))
    entries = await gateway.fetch_audit_entries(server_id, limit=1000)
    windowed = _within_window(entries, window_hours)
    channel_events = [entry for entry in windowed if _target_id(entry) == channel_id]
    by_action = Counter(str(entry.action) for entry in channel_events)

    payload = {
        "serverId": str(server_id),
        "channelId": channel_id,
        "windowHours": window_hours,
        "eventCount": len(channel_events),
        "actions": dict(by_action),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_get_incident_timeline(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments["server_id"]
    channel_id = (
        str(arguments.get("channel_id")) if arguments.get("channel_id") else None
    )
    user_id = str(arguments.get("user_id")) if arguments.get("user_id") else None
    window_hours = int(arguments["window_hours"])
    entries = await gateway.fetch_audit_entries(server_id, limit=2000)
    windowed = _within_window(entries, window_hours)

    events = []
    for entry in sorted(windowed, key=lambda value: value.created_at):
        target_id = _target_id(entry)
        actor_id = str(getattr(entry.user, "id", ""))
        if channel_id and target_id != channel_id:
            continue
        if user_id and target_id != user_id and actor_id != user_id:
            continue
        events.append(
            {
                "timestamp": entry.created_at.isoformat(),
                "source": "audit_log",
                "action": str(entry.action),
                "actorId": actor_id,
                "targetId": target_id,
                "detail": entry.reason,
            }
        )

    payload = {"events": events}
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_get_audit_actor_summary(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments["server_id"]
    window_hours = int(arguments.get("window_hours", 24))
    entries = await gateway.fetch_audit_entries(server_id, limit=2000)
    windowed = _within_window(entries, window_hours)

    actor_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"eventCount": 0, "actions": Counter()}
    )
    for entry in windowed:
        actor_id = str(getattr(entry.user, "id", "unknown"))
        actor_stats[actor_id]["eventCount"] += 1
        actor_stats[actor_id]["actions"][str(entry.action)] += 1

    actors = [
        {
            "actorId": actor_id,
            "eventCount": stats["eventCount"],
            "actions": dict(stats["actions"]),
        }
        for actor_id, stats in actor_stats.items()
    ]
    payload = {"actorCount": len(actors), "actors": actors, "windowHours": window_hours}
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_check_audit_reason_compliance(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments["server_id"]
    window_hours = int(arguments.get("window_hours", 24))
    entries = await gateway.fetch_audit_entries(server_id, limit=2000)
    windowed = _within_window(entries, window_hours)

    missing = [entry for entry in windowed if not entry.reason]
    payload = {
        "windowHours": window_hours,
        "totalChecked": len(windowed),
        "missingReasonCount": len(missing),
        "missing": [_serialize_entry(entry) for entry in missing],
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_server_health_check(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments["server_id"]
    entries = await gateway.fetch_audit_entries(server_id, limit=500)
    missing_reason = sum(1 for entry in entries if not entry.reason)
    findings = []
    if missing_reason:
        findings.append(
            {
                "severity": "medium",
                "category": "audit_reason",
                "message": f"{missing_reason} audit entries missing reason",
            }
        )

    score = max(0, 100 - min(100, missing_reason * 10))
    payload = {
        "score": score,
        "findings": findings,
        "computedAt": datetime.now(timezone.utc).isoformat(),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_governance_evidence_packager(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments["server_id"]
    window_hours = int(arguments["window_hours"])
    actor_filter = str(arguments.get("actor_id")) if arguments.get("actor_id") else None
    channel_filter = (
        str(arguments.get("channel_id")) if arguments.get("channel_id") else None
    )
    entries = await gateway.fetch_audit_entries(server_id, limit=5000)
    windowed = _within_window(entries, window_hours)

    filtered = []
    for entry in windowed:
        actor_id = str(getattr(entry.user, "id", ""))
        target_id = _target_id(entry)
        if actor_filter and actor_id != actor_filter:
            continue
        if channel_filter and target_id != channel_filter:
            continue
        filtered.append(entry)

    audit_entries = [_serialize_entry(entry) for entry in filtered]
    moderation_events = [
        entry
        for entry in audit_entries
        if entry["action"] in {"ban", "kick", "timeout"}
    ]
    channel_changes = [entry for entry in audit_entries if "channel" in entry["action"]]
    payload = {
        "bundle": {
            "auditEntries": audit_entries,
            "moderationEvents": moderation_events,
            "channelChanges": channel_changes,
        },
        "totals": {
            "auditEntries": len(audit_entries),
            "moderationEvents": len(moderation_events),
            "channelChanges": len(channel_changes),
        },
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]
