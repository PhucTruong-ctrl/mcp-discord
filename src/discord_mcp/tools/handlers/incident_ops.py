import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from mcp.types import TextContent

from discord_mcp.core.permissions import as_permission_bits, overwrite_index
from discord_mcp.core.safety import (
    build_dry_run_result,
    verify_confirm_token,
)
from discord_mcp.core.state import (
    get_channel_state,
    set_channel_state,
)

# permissions removed while a channel is locked down
LOCKDOWN_PERMISSIONS = (
    "send_messages",
    "send_messages_in_threads",
    "create_public_threads",
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _everyone_overwrite(channel: Any) -> Dict[str, Any]:
    """Current @everyone overwrite on a channel, in a restorable form."""
    everyone_id = str(
        getattr(
            getattr(getattr(channel, "guild", None), "default_role", None), "id", ""
        )
    )
    overwrite = overwrite_index(channel).get(everyone_id)
    if overwrite is None:
        return {"existed": False, "allow": 0, "deny": 0}
    allow, deny = overwrite.pair()
    return {
        "existed": True,
        "allow": as_permission_bits(allow),
        "deny": as_permission_bits(deny),
    }


async def _resolve_channels(gateway: Any, channel_ids: List[str]) -> List[Any]:
    if not gateway:
        raise ValueError("gateway is required for incident operations")
    channels = []
    for channel_id in channel_ids:
        channel = await gateway.fetch_channel(str(channel_id))
        if channel is None:
            raise ValueError(f"Channel '{channel_id}' not found")
        channels.append(channel)
    return channels


async def handle_incident_get_channel_state(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    channel_id = str(arguments["channel_id"])
    return _json({"channel_id": channel_id, "state": get_channel_state(channel_id)})


async def handle_incident_set_channel_state(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    channel_id = str(arguments["channel_id"])
    state = arguments["state"]
    if not isinstance(state, dict):
        raise ValueError("state must be an object")
    reason = _required_reason(arguments)
    action = "incident_set_channel_state"
    targets = {"channel_id": channel_id, "reason": reason}

    if bool(arguments.get("dry_run", True)):
        return _json(
            build_dry_run_result(
                action,
                targets,
                {"channelId": channel_id, "state": state, "reason": reason},
            )
        )
    verify_confirm_token(action, targets, _required_confirm_token(arguments))
    set_channel_state(channel_id, state)
    return _json({"channel_id": channel_id, "state": state, "storedAt": _now()})


async def handle_incident_apply_lockdown(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    reason = _required_reason(arguments)
    channel_ids = [str(channel_id) for channel_id in arguments["channel_ids"]]
    dry_run = bool(arguments.get("dry_run", True))
    action = "incident_apply_lockdown"
    targets = {"channel_ids": sorted(channel_ids), "reason": reason}

    if dry_run:
        payload = build_dry_run_result(
            action,
            targets,
            {
                "reason": reason,
                "channel_ids": channel_ids,
                "revokedPermissions": list(LOCKDOWN_PERMISSIONS),
            },
        )
        return _json(payload)

    confirm_token = _required_confirm_token(arguments)
    verify_confirm_token(action, targets, confirm_token)
    channels = await _resolve_channels(gateway, channel_ids)

    locked = []
    for channel in channels:
        state = get_channel_state(str(channel.id))
        if state.get("lockdown", {}).get("active"):
            previous = state["lockdown"].get("previous", {})
        else:
            previous = _everyone_overwrite(channel)
        await channel.set_permissions(
            channel.guild.default_role,
            send_messages=False,
            send_messages_in_threads=False,
            create_public_threads=False,
            reason=reason,
        )
        state["lockdown"] = {
            "active": True,
            "appliedAt": _now(),
            "reason": reason,
            "previous": previous,
            "revokedPermissions": list(LOCKDOWN_PERMISSIONS),
        }
        set_channel_state(str(channel.id), state)
        locked.append({"channelId": str(channel.id), "name": channel.name})

    return _json(
        {
            "status": "executed",
            "action": action,
            "reason": reason,
            "lockedCount": len(locked),
            "channels": locked,
        }
    )


async def handle_incident_rollback_lockdown(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps.get("gateway")
    reason = _required_reason(arguments)
    channel_ids = [str(channel_id) for channel_id in arguments["channel_ids"]]
    dry_run = bool(arguments.get("dry_run", True))
    action = "incident_rollback_lockdown"
    targets = {"channel_ids": sorted(channel_ids), "reason": reason}

    if dry_run:
        payload = build_dry_run_result(
            action,
            targets,
            {
                "reason": reason,
                "channel_ids": channel_ids,
                "restores": {
                    channel_id: get_channel_state(channel_id)
                    .get("lockdown", {})
                    .get("previous")
                    for channel_id in channel_ids
                },
            },
        )
        return _json(payload)

    confirm_token = _required_confirm_token(arguments)
    verify_confirm_token(action, targets, confirm_token)
    channels = await _resolve_channels(gateway, channel_ids)

    restored = []
    not_locked = []
    for channel in channels:
        state = get_channel_state(str(channel.id))
        lockdown = state.get("lockdown") or {}
        if not lockdown.get("active"):
            not_locked.append(str(channel.id))
            continue
        previous = lockdown.get("previous") or {}
        if previous.get("existed"):
            await channel.set_permissions(
                channel.guild.default_role,
                overwrite=_overwrite_from(previous),
                reason=reason,
            )
        else:
            # no overwrite existed before the lockdown: remove the one we added
            await channel.set_permissions(
                channel.guild.default_role, overwrite=None, reason=reason
            )
        state["lockdown"] = {
            **lockdown,
            "active": False,
            "rolledBackAt": _now(),
            "rollbackReason": reason,
        }
        set_channel_state(str(channel.id), state)
        restored.append({"channelId": str(channel.id), "restored": previous})

    return _json(
        {
            "status": "executed",
            "action": action,
            "reason": reason,
            "restoredCount": len(restored),
            "restored": restored,
            "notLocked": not_locked,
        }
    )


def _overwrite_from(previous: Dict[str, Any]):
    import discord

    return discord.PermissionOverwrite.from_pair(
        discord.Permissions(int(previous.get("allow", 0))),
        discord.Permissions(int(previous.get("deny", 0))),
    )
