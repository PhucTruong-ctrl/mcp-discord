from typing import Any, Dict, List

import discord
from mcp.types import TextContent

from discord_mcp.core.common import channel_row, json_text, require_gateway
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.validation import (
    require_reason,
    validate_enum,
    validate_snowflake,
)

_AUTO_ARCHIVE_VALUES = ["60", "1440", "4320", "10080"]
_THREAD_TYPES = {
    "public_thread": discord.ChannelType.public_thread,
    "private_thread": discord.ChannelType.private_thread,
}
_EDITABLE_FIELDS = (
    "name",
    "archived",
    "locked",
    "invitable",
    "pinned",
    "slowmode_delay",
    "auto_archive_duration",
)


def _is_dry_run(arguments: Dict[str, Any]) -> bool:
    return bool(arguments.get("dry_run", True))


def _type_name(value: Any) -> str:
    return str(getattr(value, "name", value)).lower()


def _validate_auto_archive(value: Any) -> int | None:
    if value is None:
        return None
    normalized = validate_enum(str(value), _AUTO_ARCHIVE_VALUES, "auto_archive_duration")
    return int(normalized)


def _validate_slowmode(value: Any) -> int | None:
    if value is None:
        return None
    try:
        delay = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"slowmode_delay must be an integer between 0 and 21600 seconds (got {value!r})"
        ) from None
    if not 0 <= delay <= 21600:
        raise ValueError(
            f"slowmode_delay must be between 0 and 21600 seconds (got {delay})"
        )
    return delay


def _parse_thread_type(value: Any) -> discord.ChannelType | None:
    if value is None:
        return None
    normalized = validate_enum(str(value), list(_THREAD_TYPES), "type")
    return _THREAD_TYPES[normalized]


def _resolve_parent_channel(guild: Any, channel_id: str) -> Any:
    channel_id_int = validate_snowflake(channel_id)
    channel = None
    if hasattr(guild, "get_channel"):
        channel = guild.get_channel(channel_id_int)
    if channel is None:
        for candidate in getattr(guild, "channels", []):
            if getattr(candidate, "id", None) == channel_id_int:
                channel = candidate
                break
    if channel is None:
        raise ValueError(f"Channel '{channel_id}' not found in '{guild.name}'")
    return channel


async def handle_create_thread(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_thread")
    guild = await gateway.resolve_guild(arguments["server_id"])
    channel = _resolve_parent_channel(guild, str(arguments["channel_id"]))

    if _type_name(getattr(channel, "type", None)) in ("forum", "media"):
        raise ValueError(
            f"Channel '{channel.id}' ('#{channel.name}') in '{guild.name}' is a forum "
            "channel; use the forum-post tools to create a post there instead"
        )
    if not hasattr(channel, "create_thread"):
        raise ValueError(
            f"Channel '{channel.id}' ('#{channel.name}') in '{guild.name}' does not "
            "support threads"
        )

    name = str(arguments["name"])
    auto_archive = _validate_auto_archive(arguments.get("auto_archive_duration"))
    slowmode = _validate_slowmode(arguments.get("slowmode_delay"))
    thread_type = _parse_thread_type(arguments.get("type"))
    reason = arguments.get("reason")

    action = "create_thread"
    targets = {
        "channel_id": str(arguments["channel_id"]),
        "server_id": str(arguments["server_id"]),
    }
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action, targets, {"channelId": str(channel.id), "name": name}
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    kwargs: Dict[str, Any] = {"name": name}
    if auto_archive is not None:
        kwargs["auto_archive_duration"] = auto_archive
    if slowmode is not None:
        kwargs["slowmode_delay"] = slowmode
    if thread_type is not None:
        kwargs["type"] = thread_type
    if reason is not None:
        kwargs["reason"] = reason
    thread = await channel.create_thread(**kwargs)
    return json_text(
        {"status": "executed", "action": action, "thread": channel_row(thread)}
    )


async def handle_join_thread(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "join_thread")
    thread, _guild = await gateway.resolve_thread(
        str(arguments["thread_id"]), arguments["server_id"]
    )
    await thread.join()
    return json_text(
        {"status": "executed", "action": "join_thread", "threadId": str(thread.id)}
    )


async def handle_leave_thread(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "leave_thread")
    thread, _guild = await gateway.resolve_thread(
        str(arguments["thread_id"]), arguments["server_id"]
    )
    await thread.leave()
    return json_text(
        {"status": "executed", "action": "leave_thread", "threadId": str(thread.id)}
    )


async def handle_add_thread_member(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "add_thread_member")
    thread, _guild = await gateway.resolve_thread(
        str(arguments["thread_id"]), arguments["server_id"]
    )
    member = await gateway.resolve_member(
        str(arguments["user_id"]), arguments["server_id"]
    )

    action = "add_thread_member"
    targets = {
        "server_id": str(arguments["server_id"]),
        "thread_id": str(arguments["thread_id"]),
        "user_id": str(arguments["user_id"]),
    }
    details = {"threadId": str(thread.id), "userId": str(member.id)}
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await thread.add_user(member)
    return json_text(
        {"status": "executed", "action": action, **details}
    )


async def handle_remove_thread_member(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "remove_thread_member")
    thread, _guild = await gateway.resolve_thread(
        str(arguments["thread_id"]), arguments["server_id"]
    )
    member = await gateway.resolve_member(
        str(arguments["user_id"]), arguments["server_id"]
    )

    action = "remove_thread_member"
    targets = {
        "server_id": str(arguments["server_id"]),
        "thread_id": str(arguments["thread_id"]),
        "user_id": str(arguments["user_id"]),
    }
    details = {"threadId": str(thread.id), "userId": str(member.id)}
    if _is_dry_run(arguments):
        return json_text(build_dry_run_result(action, targets, details))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await thread.remove_user(member)
    return json_text(
        {"status": "executed", "action": action, **details}
    )


async def handle_edit_thread(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "edit_thread")
    thread, _guild = await gateway.resolve_thread(
        str(arguments["thread_id"]), arguments["server_id"]
    )

    updates: Dict[str, Any] = {}
    if arguments.get("name") is not None:
        updates["name"] = str(arguments["name"])
    for flag in ("archived", "locked", "invitable", "pinned"):
        if arguments.get(flag) is not None:
            updates[flag] = arguments[flag]
    slowmode = _validate_slowmode(arguments.get("slowmode_delay"))
    if slowmode is not None:
        updates["slowmode_delay"] = slowmode
    auto_archive = _validate_auto_archive(arguments.get("auto_archive_duration"))
    if auto_archive is not None:
        updates["auto_archive_duration"] = auto_archive
    reason = arguments.get("reason")

    if not updates:
        raise ValueError(
            "edit_thread requires at least one of: "
            + ", ".join(_EDITABLE_FIELDS)
        )

    action = "edit_thread"
    targets = {
        "server_id": str(arguments["server_id"]),
        "thread_id": str(arguments["thread_id"]),
    }
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {"threadId": str(thread.id), "updates": sorted(updates)},
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    if reason is not None:
        updates["reason"] = reason
    edited = await thread.edit(**updates)
    return json_text(
        {
            "status": "executed",
            "action": action,
            "threadId": str(thread.id),
            "name": edited.name,
            "archived": edited.archived,
            "locked": edited.locked,
            "invitable": edited.invitable,
            "slowmodeDelay": edited.slowmode_delay,
        }
    )


async def handle_delete_thread(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "delete_thread")
    thread, _guild = await gateway.resolve_thread(
        str(arguments["thread_id"]), arguments["server_id"]
    )

    action = "delete_thread"
    reason = require_reason(arguments.get("reason"), action)
    targets = {
        "server_id": str(arguments["server_id"]),
        "thread_id": str(arguments["thread_id"]),
    }
    if _is_dry_run(arguments):
        return json_text(
            build_dry_run_result(
                action, targets, {"threadId": str(thread.id), "name": thread.name}
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    await thread.delete(reason=reason)
    return json_text(
        {"status": "executed", "action": action, "threadId": str(thread.id)}
    )


async def handle_list_active_threads(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "list_active_threads")
    guild = await gateway.resolve_guild(arguments["server_id"])
    threads = await guild.active_threads()
    rows = [channel_row(thread) for thread in sorted(threads, key=lambda t: t.id)]
    return json_text(
        {"serverId": str(guild.id), "count": len(rows), "threads": rows}
    )
