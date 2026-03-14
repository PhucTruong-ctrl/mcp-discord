import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List

from mcp.types import TextContent


def _serialize_entry(entry: Any) -> Dict[str, Any]:
    return {
        "action": str(entry.action),
        "actorId": str(entry.user.id) if getattr(entry, "user", None) else None,
        "targetId": str(entry.target.id) if getattr(entry, "target", None) else None,
        "reason": entry.reason,
        "timestamp": entry.created_at.isoformat(),
    }


def _within_window(entries: Iterable[Any], window_hours: int) -> List[Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    return [entry for entry in entries if entry.created_at >= cutoff]


async def handle_get_audit_log(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments["server_id"]
    limit = int(arguments.get("limit", 50))
    action_type = arguments.get("action_type")
    entries = await gateway.fetch_audit_entries(
        server_id, limit=limit, action_type=action_type
    )

    payload = {
        "serverId": str(server_id),
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
    server_id = arguments["server_id"]
    user_id = str(arguments["user_id"])
    limit = int(arguments.get("limit", 200))
    entries = await gateway.fetch_audit_entries(server_id, limit=limit)

    history = [
        entry for entry in entries if str(getattr(entry.target, "id", "")) == user_id
    ]
    payload = {
        "serverId": str(server_id),
        "targetUserId": user_id,
        "eventCount": len(history),
        "events": [_serialize_entry(entry) for entry in history],
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
    channel_events = [
        e for e in windowed if str(getattr(e.target, "id", "")) == channel_id
    ]
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
        target_id = str(getattr(entry.target, "id", ""))
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
        target_id = str(getattr(entry.target, "id", ""))
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
