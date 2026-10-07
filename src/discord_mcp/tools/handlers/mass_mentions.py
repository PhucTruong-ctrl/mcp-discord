import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from mcp.types import TextContent

from discord_mcp.core.permissions import (
    as_permission_bits,
    effective_permissions,
    role_payload,
)

MENTION_EVERYONE_BIT = 1 << 17
# channels that expose history(); forum channels are post containers and are covered
# through their threads instead
_HISTORY_CHANNEL_TYPES = ("text", "news", "public_thread", "private_thread")
_THREAD_PARENT_TYPES = _HISTORY_CHANNEL_TYPES + ("forum",)
_MASS_MENTION_TOKENS = ("@everyone", "@here")


def _channel_type(channel: Any) -> str:
    return str(getattr(channel, "type", ""))


def _type_matches(channel: Any, candidates: tuple) -> bool:
    return any(token in _channel_type(channel) for token in candidates)


async def _iter_archived(channel: Any) -> List[Any]:
    archived: List[Any] = []
    iterator = getattr(channel, "archived_threads", None)
    if iterator is None:
        return archived
    try:
        async for thread in iterator(limit=100):
            archived.append(thread)
    except Exception:  # noqa: BLE001 - archived listing is best effort
        return archived
    return archived


def _hit_payload(
    message: Any, channel: Any, guild: Any, cache: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    author = getattr(message, "author", None)
    author_id = str(getattr(author, "id", ""))
    key = f"{author_id}:{channel.id}"
    if key not in cache:
        member = guild.get_member(int(author_id)) if author_id.isdigit() else None
        granting: List[Dict[str, Any]] = []
        channel_allows = False
        if member is not None:
            granting = [
                role_payload(role)
                for role in getattr(member, "roles", []) or []
                if as_permission_bits(getattr(role, "permissions", 0))
                & MENTION_EVERYONE_BIT
            ]
            try:
                resolved = effective_permissions(guild, member, channel)
                channel_allows = bool(resolved["effective"] & MENTION_EVERYONE_BIT)
            except Exception:  # noqa: BLE001 - permission context is best effort
                channel_allows = False
        cache[key] = {
            "authorHasMentionEveryoneNow": bool(granting) or channel_allows,
            "authorGrantingRoles": granting,
            "channelAllowsEveryone": channel_allows,
        }

    content = message.content or ""
    return {
        "messageId": str(message.id),
        "channelId": str(channel.id),
        "channelName": getattr(channel, "name", str(channel.id)),
        "authorId": author_id,
        "authorName": str(author),
        "timestamp": message.created_at.isoformat(),
        "mentionEveryone": bool(getattr(message, "mention_everyone", False)),
        "kind": (
            "delivered"
            if getattr(message, "mention_everyone", False)
            else "suppressed_text"
        ),
        "content": content[:200],
        "mentions": [
            {"id": str(user.id), "name": str(user)}
            for user in (getattr(message, "mentions", None) or [])
        ],
        "roleMentionIds": [
            str(role.id) for role in (getattr(message, "role_mentions", None) or [])
        ],
        **cache[key],
    }


async def handle_audit_mass_mentions(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    guild = await gateway.resolve_guild(arguments["server_id"])
    filter_ids = {str(v) for v in arguments.get("channel_ids", []) or []}
    window_hours = float(arguments.get("window_hours", 168) or 0)
    scan_limit = min(int(arguments.get("scan_limit", 200) or 200), 1000)
    include_threads = bool(arguments.get("include_threads", False))
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=window_hours)
        if window_hours > 0
        else None
    )

    hits: List[Dict[str, Any]] = []
    access_errors: List[Dict[str, str]] = []
    permission_cache: Dict[str, Dict[str, Any]] = {}
    scanned_channels = 0
    scanned_messages = 0

    targets: List[Any] = []
    for channel in getattr(guild, "channels", []) or []:
        if filter_ids and str(channel.id) not in filter_ids:
            continue
        if _type_matches(channel, _HISTORY_CHANNEL_TYPES):
            targets.append(channel)
        if include_threads and _type_matches(channel, _THREAD_PARENT_TYPES):
            targets.extend(await _iter_archived(channel))

    for channel in targets:
        scanned_channels += 1
        try:
            async for message in channel.history(limit=scan_limit):
                scanned_messages += 1
                created_at = getattr(message, "created_at", None)
                if (
                    cutoff is not None
                    and created_at is not None
                    and created_at < cutoff
                ):
                    break
                content = message.content or ""
                if getattr(message, "mention_everyone", False) or any(
                    token in content for token in _MASS_MENTION_TOKENS
                ):
                    hits.append(_hit_payload(message, channel, guild, permission_cache))
        except Exception as exc:  # noqa: BLE001 - one unreadable channel must not fail all
            access_errors.append(
                {
                    "channelId": str(getattr(channel, "id", "")),
                    "channelName": str(getattr(channel, "name", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    hits.sort(key=lambda hit: hit["timestamp"])
    delivered = [hit for hit in hits if hit["kind"] == "delivered"]
    payload = {
        "serverId": str(guild.id),
        "windowHours": window_hours,
        "scanLimit": scan_limit,
        "includeThreads": include_threads,
        "scannedChannels": scanned_channels,
        "scannedMessages": scanned_messages,
        "hitCount": len(hits),
        "deliveredCount": len(delivered),
        "suppressedCount": len(hits) - len(delivered),
        "hits": hits,
        "accessErrors": access_errors,
        "note": (
            "kind=delivered means Discord registered a mass mention (mention_everyone="
            "true) and members were notified. kind=suppressed_text means the message only "
            "contains '@everyone'/'@here' as literal text: the author lacked "
            "MENTION_EVERYONE, so no notification was sent."
        ),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]
