"""Advanced message tools: attachments, components, polls, forwards, pins, reactions, threads.

Every handler is gateway-dependent and starts with ``require_gateway``; snowflakes cross the
MCP boundary as strings; entity misses raise ``ValueError`` naming the id and the server.
"""

from __future__ import annotations

import io
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import discord
from mcp.types import TextContent

from discord_mcp.core.common import fetch_bytes, json_text, require_gateway, user_row
from discord_mcp.core.emoji import emoji_payload, emoji_token, parse_emoji, serialize_emoji
from discord_mcp.core.resolve import try_int
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token

_AUTO_ARCHIVE_DURATIONS = (60, 1440, 4320, 10080)
_MAX_SLOWMODE_SECONDS = 21600


def _snowflake(value: Any, field: str, server_id: str) -> int:
    parsed = try_int(value)
    if parsed is None:
        raise ValueError(f"{field} '{value}' is not a valid snowflake on server '{server_id}'")
    return parsed


async def _fetch_channel(gateway: Any, channel_id: Any, server_id: str):
    cid = _snowflake(channel_id, "channel_id", server_id)
    try:
        channel = await gateway.fetch_channel(str(cid))
    except discord.NotFound as exc:
        raise ValueError(f"channel '{channel_id}' not found on server '{server_id}'") from exc
    if channel is None:
        raise ValueError(f"channel '{channel_id}' not found on server '{server_id}'")
    return channel


async def _fetch_message(gateway: Any, channel_id: Any, message_id: Any, server_id: str):
    channel = await _fetch_channel(gateway, channel_id, server_id)
    mid = _snowflake(message_id, "message_id", server_id)
    missing = ValueError(
        f"message '{message_id}' in channel '{channel_id}' on server '{server_id}' not found"
    )
    try:
        message = await channel.fetch_message(mid)
    except discord.NotFound as exc:
        raise missing from exc
    if message is None:
        raise missing
    return channel, message


def _as_list(value: Any, field: str) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"{field} must be a list")


def _send_payload(action: str, channel: Any, message: Any) -> Dict[str, Any]:
    return {
        "status": "executed",
        "action": action,
        "messageId": str(message.id),
        "channelId": str(channel.id),
        "attachments": [
            {"filename": a.filename, "size": a.size, "url": a.url}
            for a in getattr(message, "attachments", None) or []
        ],
    }


# --- files -----------------------------------------------------------------


def _filename_from_url(url: str) -> str:
    return os.path.basename(urlparse(url).path) or "download"




async def _load_files(file_paths: Any, file_urls: Any) -> List[discord.File]:
    files: List[discord.File] = []
    for path in _as_list(file_paths, "file_paths"):
        try:
            files.append(discord.File(str(path)))
        except OSError as exc:
            raise ValueError(f"file_paths: cannot read '{path}': {exc}") from exc
    for url in _as_list(file_urls, "file_urls"):
        data = await fetch_bytes(str(url), "file_urls")
        files.append(discord.File(io.BytesIO(data), filename=_filename_from_url(str(url))))
    return files


# --- mentions --------------------------------------------------------------


def _mention_ids(value: Any, field: str) -> Any:
    if isinstance(value, bool):
        return value
    ids = []
    for raw in _as_list(value, field):
        parsed = try_int(raw)
        if parsed is None:
            raise ValueError(f"{field}: '{raw}' is not a valid snowflake")
        ids.append(parsed)
    return ids


def _allowed_mentions(arguments: Dict[str, Any]) -> Optional[discord.AllowedMentions]:
    kwargs: Dict[str, Any] = {}
    if "mention_everyone" in arguments:
        kwargs["everyone"] = bool(arguments["mention_everyone"])
    if "allowed_mention_roles" in arguments:
        kwargs["roles"] = _mention_ids(arguments["allowed_mention_roles"], "allowed_mention_roles")
    if "allowed_mention_users" in arguments:
        kwargs["users"] = _mention_ids(arguments["allowed_mention_users"], "allowed_mention_users")
    if "allowed_mention_echo" in arguments:
        kwargs["replied_user"] = bool(arguments["allowed_mention_echo"])
    if not kwargs:
        return None
    return discord.AllowedMentions(**kwargs)


def _sticker_items(sticker_ids: Any, server_id: str) -> Optional[List[discord.Object]]:
    ids = _as_list(sticker_ids, "sticker_ids")
    if not ids:
        return None
    return [discord.Object(id=_snowflake(raw, "sticker_ids", server_id)) for raw in ids]


# --- components ------------------------------------------------------------


def _build_button(spec: Dict[str, Any], index: int) -> discord.ui.Button:
    raw_style = spec.get("style", 1)
    try:
        style_value = int(raw_style)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"components[{index}].style must be an integer 1-5, got {raw_style!r}"
        ) from exc
    if not 1 <= style_value <= 5:
        raise ValueError(
            f"components[{index}].style must be between 1 and 5 "
            f"(1 primary, 2 secondary, 3 success, 4 danger, 5 link), got {raw_style!r}"
        )
    kwargs: Dict[str, Any] = {"style": discord.ButtonStyle(style_value)}
    for key in ("custom_id", "label", "emoji", "url", "disabled", "row"):
        if spec.get(key) is not None:
            kwargs[key] = spec[key]
    return discord.ui.Button(**kwargs)


def _build_select(spec: Dict[str, Any], index: int) -> discord.ui.Select:
    custom_id = spec.get("custom_id")
    if not custom_id:
        raise ValueError(f"components[{index}]: select requires custom_id")
    option_specs = _as_list(spec.get("options"), f"components[{index}].options")
    if not option_specs:
        raise ValueError(f"components[{index}]: select requires at least one option")
    options = []
    for opt_index, option in enumerate(option_specs):
        if not isinstance(option, dict):
            raise ValueError(f"components[{index}].options[{opt_index}] must be an object")
        label = option.get("label")
        if not label:
            raise ValueError(f"components[{index}].options[{opt_index}]: label is required")
        kwargs: Dict[str, Any] = {"label": label}
        if option.get("value") is not None:
            kwargs["value"] = str(option["value"])
        for key in ("description", "emoji", "default"):
            if option.get(key) is not None:
                kwargs[key] = option[key]
        options.append(discord.SelectOption(**kwargs))
    kwargs = {"custom_id": str(custom_id), "options": options}
    for key in ("placeholder", "min_values", "max_values", "disabled", "required", "row"):
        if spec.get(key) is not None:
            kwargs[key] = spec[key]
    return discord.ui.Select(**kwargs)


def _build_view(specs: List[Any]) -> discord.ui.View:
    view = discord.ui.View()
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"components[{index}] must be an object")
        kind = str(spec.get("type", "")).strip().lower()
        if kind == "button":
            view.add_item(_build_button(spec, index))
        elif kind == "select":
            view.add_item(_build_select(spec, index))
        else:
            raise ValueError(
                f"components[{index}]: unknown component type {spec.get('type')!r}; "
                "expected 'button' or 'select'"
            )
    return view


# --- polls -----------------------------------------------------------------


def _poll_hours(value: Any) -> int:
    try:
        hours = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"send_poll: duration_hours must be an integer, got {value!r}") from exc
    if not 1 <= hours <= 768:
        raise ValueError(
            f"send_poll: duration_hours must be between 1 and 768 (Discord limit), got {hours}"
        )
    return hours


def _poll_layout(value: Any) -> Optional[discord.PollLayoutType]:
    if value is None:
        return None
    text = str(value).strip()
    for member in discord.PollLayoutType:
        if member.name == text.lower() or str(member.value) == text:
            return member
    allowed = ", ".join(member.name for member in discord.PollLayoutType)
    raise ValueError(f"send_poll: layout_type must be one of: {allowed} (got {value!r})")


def _poll_answer(entry: Any, index: int) -> Tuple[str, Optional[str]]:
    if isinstance(entry, str):
        text, raw_emoji = entry, None
    elif isinstance(entry, dict):
        text = str(entry.get("text") or "")
        raw_emoji = entry.get("emoji")
    else:
        raise ValueError(
            f"send_poll: answers[{index}] must be a string or an object with a text field"
        )
    if not text.strip():
        raise ValueError(f"send_poll: answers[{index}]: text is required")
    # core.emoji codec: unicode text, <name:id> tokens and {emoji, emojiId} shapes all round-trip
    emoji = parse_emoji(None, raw_emoji) if raw_emoji is not None else None
    return text.strip(), emoji


# --- reactions -------------------------------------------------------------


def _emoji_key(name: Optional[str], emoji_id: Optional[str]) -> Tuple[str, str]:
    if emoji_id:
        return ("id", str(emoji_id))
    return ("text", str(name))


def _requested_emoji_key(token: Optional[str]) -> Tuple[str, str]:
    text = str(token or "")
    if text.startswith("<") and text.endswith(">"):
        inner = text[1:-1].split(":")
        if len(inner) == 3 and try_int(inner[2]) is not None:
            return ("id", str(try_int(inner[2])))
    return ("text", text)


def _reaction_token(reaction: Any) -> str:
    name, emoji_id, animated = serialize_emoji(reaction.emoji)
    return emoji_token(name, emoji_id, animated) or ""


def _reaction_limit(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"limit must be an integer, got {value!r}") from exc
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    return limit


# --- handlers --------------------------------------------------------------


async def handle_send_message_with_files(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "send_message_with_files")
    server_id = str(arguments["server_id"])
    channel = await _fetch_channel(gateway, arguments["channel_id"], server_id)
    files = await _load_files(arguments.get("file_paths"), arguments.get("file_urls"))
    stickers = _sticker_items(arguments.get("sticker_ids"), server_id)

    content = arguments.get("content")
    if not content and not files and not stickers:
        raise ValueError(
            "send_message_with_files: content, file_paths, file_urls or sticker_ids is "
            f"required (server '{server_id}', channel '{channel.id}')"
        )

    kwargs: Dict[str, Any] = {
        "tts": bool(arguments.get("tts", False)),
        "silent": bool(arguments.get("silent", False)),
        "suppress_embeds": bool(arguments.get("suppress_embeds", False)),
    }
    if content is not None:
        kwargs["content"] = str(content)
    if files:
        kwargs["files"] = files
    if stickers:
        kwargs["stickers"] = stickers
    allowed_mentions = _allowed_mentions(arguments)
    if allowed_mentions is not None:
        kwargs["allowed_mentions"] = allowed_mentions
    if "mention_author" in arguments:
        kwargs["mention_author"] = bool(arguments["mention_author"])
    if arguments.get("nonce") is not None:
        kwargs["nonce"] = arguments["nonce"]
    if arguments.get("delete_after") is not None:
        kwargs["delete_after"] = float(arguments["delete_after"])

    message = await channel.send(**kwargs)
    return json_text(_send_payload("send_message_with_files", channel, message))


async def handle_send_components(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "send_components")
    server_id = str(arguments["server_id"])
    specs = _as_list(arguments["components"], "components")
    if not specs:
        raise ValueError("components must contain at least one component spec")
    view = _build_view(specs)
    channel = await _fetch_channel(gateway, arguments["channel_id"], server_id)
    message = await channel.send(view=view)
    return json_text(_send_payload("send_components", channel, message))


async def handle_send_poll(arguments: Dict[str, Any], deps: Dict[str, Any]) -> List[TextContent]:
    gateway = require_gateway(deps, "send_poll")
    server_id = str(arguments["server_id"])
    question = str(arguments.get("question") or "").strip()
    if not question:
        raise ValueError(f"send_poll: question is required (server '{server_id}')")
    answer_entries = _as_list(arguments["answers"], "answers")
    if not 2 <= len(answer_entries) <= 10:
        raise ValueError(
            f"send_poll: answers must have 2-10 entries, got {len(answer_entries)} "
            f"(server '{server_id}')"
        )
    hours = _poll_hours(arguments["duration_hours"])
    layout = _poll_layout(arguments.get("layout_type"))

    poll_kwargs: Dict[str, Any] = {"multiple": bool(arguments.get("multiple", False))}
    if layout is not None:
        poll_kwargs["layout_type"] = layout
    poll = discord.Poll(question, timedelta(hours=hours), **poll_kwargs)
    for index, entry in enumerate(answer_entries):
        text, emoji = _poll_answer(entry, index)
        poll.add_answer(text=text, emoji=emoji)

    channel = await _fetch_channel(gateway, arguments["channel_id"], server_id)
    message = await channel.send(poll=poll)
    return json_text(_send_payload("send_poll", channel, message))


async def handle_get_poll_results(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_poll_results")
    server_id = str(arguments["server_id"])
    channel, message = await _fetch_message(
        gateway, arguments["channel_id"], arguments["message_id"], server_id
    )
    poll = getattr(message, "poll", None)
    if poll is None:
        raise ValueError(
            f"message '{arguments['message_id']}' in channel '{channel.id}' on server "
            f"'{server_id}' has no poll"
        )
    finalized = bool(poll.is_finalized())
    total_votes = poll.total_votes
    if total_votes is None:
        total_votes = sum(getattr(answer, "vote_count", 0) for answer in poll.answers)
    answers = []
    for answer in poll.answers:
        media = answer.media
        answers.append(
            {
                "answerId": str(answer.id),
                "text": media.text,
                "partial": not finalized,
                "pollMedia": {"text": media.text, "emoji": emoji_payload(media.emoji)},
            }
        )
    return json_text(
        {
            "messageId": str(message.id),
            "question": poll.question,
            "multiple": bool(poll.multiple),
            "durationHours": poll.duration.total_seconds() / 3600,
            "totalVotes": int(total_votes),
            "answers": answers,
            "finalized": finalized,
        }
    )


async def handle_forward_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "forward_message")
    server_id = str(arguments["server_id"])
    channel, message = await _fetch_message(
        gateway, arguments["channel_id"], arguments["message_id"], server_id
    )
    destination = await _fetch_channel(
        gateway, arguments["destination_channel_id"], server_id
    )
    forwarded = await message.forward(destination, fail_if_not_exists=True)
    return json_text(
        {
            "status": "executed",
            "action": "forward_message",
            "messageId": str(message.id),
            "channelId": str(channel.id),
            "destinationChannelId": str(destination.id),
            "forwardedMessageId": str(forwarded.id),
        }
    )


async def handle_pin_message(arguments: Dict[str, Any], deps: Dict[str, Any]) -> List[TextContent]:
    gateway = require_gateway(deps, "pin_message")
    server_id = str(arguments["server_id"])
    channel_id = str(arguments["channel_id"])
    message_id = str(arguments["message_id"])
    reason = arguments.get("reason")
    action = "pin_message"
    targets = {"channel_id": channel_id, "message_id": message_id}
    if bool(arguments.get("dry_run", True)):
        return json_text(build_dry_run_result(action, targets, {"reason": reason or ""}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    _, message = await _fetch_message(
        gateway, arguments["channel_id"], arguments["message_id"], server_id
    )
    await message.pin(reason=reason)
    return json_text(
        {"status": "executed", "action": action, "messageId": message_id, "channelId": channel_id}
    )


async def handle_unpin_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "unpin_message")
    server_id = str(arguments["server_id"])
    channel_id = str(arguments["channel_id"])
    message_id = str(arguments["message_id"])
    reason = arguments.get("reason")
    action = "unpin_message"
    targets = {"channel_id": channel_id, "message_id": message_id}
    if bool(arguments.get("dry_run", True)):
        return json_text(build_dry_run_result(action, targets, {"reason": reason or ""}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    _, message = await _fetch_message(
        gateway, arguments["channel_id"], arguments["message_id"], server_id
    )
    await message.unpin(reason=reason)
    return json_text(
        {"status": "executed", "action": action, "messageId": message_id, "channelId": channel_id}
    )


async def handle_clear_message_reactions(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "clear_message_reactions")
    server_id = str(arguments["server_id"])
    channel_id = str(arguments["channel_id"])
    message_id = str(arguments["message_id"])
    reason = arguments.get("reason")
    action = "clear_message_reactions"
    targets = {"channel_id": channel_id, "message_id": message_id}
    if bool(arguments.get("dry_run", True)):
        return json_text(build_dry_run_result(action, targets, {"reason": reason or ""}))
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    _, message = await _fetch_message(
        gateway, arguments["channel_id"], arguments["message_id"], server_id
    )
    # Discord's clear-all-reactions endpoint takes no reason parameter
    await message.clear_reactions()
    return json_text(
        {"status": "executed", "action": action, "messageId": message_id, "channelId": channel_id}
    )


async def handle_get_reaction_users(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "get_reaction_users")
    server_id = str(arguments["server_id"])
    message_id = str(arguments["message_id"])
    guild = await gateway.resolve_guild(str(arguments["server_id"]))
    channel, message = await _fetch_message(
        gateway, arguments["channel_id"], arguments["message_id"], server_id
    )
    reactions = list(getattr(message, "reactions", None) or [])
    if not reactions:
        raise ValueError(
            f"message '{message_id}' in channel '{channel.id}' on server '{server_id}' "
            "has no reactions"
        )

    token = parse_emoji(guild, arguments["emoji"])
    wanted = _requested_emoji_key(token)
    reaction = next(
        (r for r in reactions if _emoji_key(*serialize_emoji(r.emoji)[:2]) == wanted),
        None,
    )
    if reaction is None:
        available = ", ".join(sorted({_reaction_token(r) for r in reactions}))
        raise ValueError(
            f"emoji '{arguments['emoji']}' not found on message '{message_id}' in channel "
            f"'{channel.id}' on server '{server_id}'; available: {available}"
        )

    limit = _reaction_limit(arguments.get("limit"))
    users = [user_row(user) async for user in reaction.users(limit=limit)]
    return json_text(
        {
            "messageId": str(message.id),
            "emoji": token,
            "count": int(reaction.count),
            "users": users,
        }
    )


async def handle_create_thread_from_message(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "create_thread_from_message")
    server_id = str(arguments["server_id"])
    channel_id = str(arguments["channel_id"])
    message_id = str(arguments["message_id"])
    name = str(arguments.get("name") or "").strip()
    if not name:
        raise ValueError(f"name is required for create_thread_from_message (server '{server_id}')")
    reason = arguments.get("reason")

    auto_archive = arguments.get("auto_archive_duration")
    if auto_archive is not None:
        parsed = try_int(auto_archive)
        if parsed not in _AUTO_ARCHIVE_DURATIONS:
            raise ValueError(
                "auto_archive_duration must be one of "
                f"{', '.join(str(v) for v in _AUTO_ARCHIVE_DURATIONS)}, got {auto_archive!r}"
            )
        auto_archive = parsed

    slowmode = arguments.get("slowmode_delay")
    if slowmode is not None:
        parsed = try_int(slowmode)
        if parsed is None or not 0 <= parsed <= _MAX_SLOWMODE_SECONDS:
            raise ValueError(
                f"slowmode_delay must be between 0 and {_MAX_SLOWMODE_SECONDS} seconds, "
                f"got {slowmode!r}"
            )
        slowmode = parsed

    action = "create_thread_from_message"
    targets = {"channel_id": channel_id, "message_id": message_id, "name": name}
    if bool(arguments.get("dry_run", True)):
        return json_text(
            build_dry_run_result(
                action,
                targets,
                {
                    "reason": reason or "",
                    "autoArchiveDuration": auto_archive,
                    "slowmodeDelay": slowmode,
                },
            )
        )
    verify_confirm_token(action, targets, arguments.get("confirm_token"))

    _, message = await _fetch_message(
        gateway, arguments["channel_id"], arguments["message_id"], server_id
    )
    thread = await message.create_thread(
        name=name,
        auto_archive_duration=auto_archive,
        slowmode_delay=slowmode,
        reason=reason,
    )
    return json_text(
        {
            "status": "executed",
            "action": action,
            "messageId": message_id,
            "channelId": channel_id,
            "threadId": str(thread.id),
            "threadName": thread.name,
        }
    )


async def handle_send_typing(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = require_gateway(deps, "send_typing")
    server_id = str(arguments["server_id"])
    channel = await _fetch_channel(gateway, arguments["channel_id"], server_id)
    typing = getattr(channel, "typing", None)
    if typing is None:
        raise ValueError(
            f"channel '{channel.id}' on server '{server_id}' does not support typing "
            "(text channels and threads only)"
        )
    async with typing():
        pass
    return json_text(
        {"status": "executed", "action": "send_typing", "channelId": str(channel.id)}
    )
