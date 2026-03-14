import json
from typing import Any, Dict, List

import discord
from mcp.types import TextContent


def _resolve_post_id(arguments: Dict[str, Any]) -> str:
    post_id = arguments.get("post_id") or arguments.get("postId")
    if not post_id:
        raise ValueError("post_id is required")
    return str(post_id)


def _history_limit(arguments: Dict[str, Any], key: str, default: int = 50) -> int:
    return min(int(arguments.get(key, default)), 100)


async def _read_post_messages(
    thread: Any,
    serialize_message: Any,
    limit: int,
    before_obj: Any = None,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    async for message in thread.history(limit=limit, before=before_obj):
        payload = serialize_message(message)
        payload.update({"authorId": str(message.author.id)})
        messages.append(payload)
    return messages


def _resolve_tag_names(arguments: Dict[str, Any]) -> List[str]:
    tag_names = arguments.get("tag_names") or arguments.get("tagNames")
    if not isinstance(tag_names, list) or not tag_names:
        raise ValueError("tag_names must be a non-empty array")
    return [str(name) for name in tag_names]


def _lookup_tags(
    forum_channel: Any,
    normalize_name: Any,
    tag_names: List[str],
) -> List[Any]:
    available_by_name = {
        normalize_name(tag.name): tag for tag in forum_channel.available_tags
    }
    selected = []
    missing = []
    for raw_name in tag_names:
        key = normalize_name(raw_name)
        tag = available_by_name.get(key)
        if tag is None:
            missing.append(raw_name)
            continue
        selected.append(tag)
    if missing:
        available = ", ".join(tag.name for tag in forum_channel.available_tags)
        raise ValueError(
            f"Tags not found: {', '.join(missing)}. Available tags: {available}"
        )
    return selected


async def handle_list_forum_posts(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    serialize_forum_tag = deps["serialize_forum_tag"]
    max_archived_threads_scan = deps["max_archived_threads_scan"]

    server_id = arguments.get("server_id") or arguments.get("server")
    channel_identifier = arguments["channel"]
    limit = _history_limit(arguments, "limit")
    include_archived = bool(
        arguments.get("include_archived", arguments.get("includeArchived", False))
    )

    forum_channel = await gateway.resolve_forum_channel(channel_identifier, server_id)
    threads = await gateway.collect_forum_threads(
        forum_channel, include_archived, max_archived_threads_scan
    )
    posts = sorted(
        threads,
        key=lambda thread: thread.created_at or discord.utils.utcnow(),
        reverse=True,
    )[:limit]

    payload = {
        "forumChannel": forum_channel.name,
        "server": forum_channel.guild.name,
        "includeArchived": include_archived,
        "totalPosts": len(posts),
        "posts": [
            {
                "postId": str(post.id),
                "postName": post.name,
                "createdAt": post.created_at.isoformat() if post.created_at else None,
                "archived": post.archived,
                "locked": post.locked,
                "messageCount": post.message_count,
                "memberCount": post.member_count,
                "tags": [serialize_forum_tag(tag) for tag in post.applied_tags],
            }
            for post in posts
        ],
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_read_forum_post_messages(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    serialize_message = deps["serialize_message"]
    try_int = deps["try_int"]

    server_id = arguments.get("server_id") or arguments.get("server")
    post_id = _resolve_post_id(arguments)
    limit = _history_limit(arguments, "limit")
    before = arguments.get("before")
    before_obj = discord.Object(id=int(before)) if try_int(before) else None

    thread, guild = await gateway.resolve_forum_post(post_id, server_id)
    messages = await _read_post_messages(thread, serialize_message, limit, before_obj)

    payload = {
        "postId": str(thread.id),
        "postName": thread.name,
        "server": guild.name,
        "messageCount": len(messages),
        "messages": messages,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_read_forum_posts_batch(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    serialize_message = deps["serialize_message"]

    server_id = arguments.get("server_id") or arguments.get("server")
    post_ids = arguments.get("post_ids") or arguments.get("postIds")
    if not isinstance(post_ids, list) or not post_ids:
        raise ValueError("post_ids must be a non-empty array")

    limit_per_post = _history_limit(arguments, "limit_per_post", default=10)

    results = []
    for raw_post_id in post_ids:
        post_id = str(raw_post_id)
        thread, _ = await gateway.resolve_forum_post(post_id, server_id)
        messages = await _read_post_messages(thread, serialize_message, limit_per_post)
        results.append(
            {
                "postId": str(thread.id),
                "postName": thread.name,
                "messageCount": len(messages),
                "messages": messages,
            }
        )

    payload = {
        "totalPosts": len(results),
        "limitPerPost": limit_per_post,
        "results": results,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_get_thread_context(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    serialize_message = deps["serialize_message"]

    server_id = arguments.get("server_id") or arguments.get("server")
    post_id = _resolve_post_id(arguments)
    thread, guild = await gateway.resolve_forum_post(post_id, server_id)
    messages = await _read_post_messages(thread, serialize_message, 3)

    payload = {
        "post": {
            "postId": str(thread.id),
            "postName": thread.name,
            "server": guild.name,
        },
        "starterMessage": messages[0] if messages else None,
        "recentMessages": messages,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_list_thread_participants(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]

    server_id = arguments.get("server_id") or arguments.get("server")
    post_id = _resolve_post_id(arguments)
    thread, _ = await gateway.resolve_forum_post(post_id, server_id)

    participants: Dict[str, Dict[str, Any]] = {}
    async for message in thread.history(limit=100):
        user_id = str(message.author.id)
        participants[user_id] = {"userId": user_id, "username": str(message.author)}

    payload = {
        "postId": str(thread.id),
        "postName": thread.name,
        "participantCount": len(participants),
        "participants": sorted(participants.values(), key=lambda item: item["userId"]),
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_get_thread_activity_summary(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]

    server_id = arguments.get("server_id") or arguments.get("server")
    post_id = _resolve_post_id(arguments)
    thread, _ = await gateway.resolve_forum_post(post_id, server_id)

    message_count = 0
    participant_ids = set()
    first_message_at = None
    last_message_at = None

    async for message in thread.history(limit=100):
        message_count += 1
        participant_ids.add(str(message.author.id))
        if first_message_at is None or message.created_at < first_message_at:
            first_message_at = message.created_at
        if last_message_at is None or message.created_at > last_message_at:
            last_message_at = message.created_at

    payload = {
        "postId": str(thread.id),
        "postName": thread.name,
        "messageCount": message_count,
        "uniqueParticipantCount": len(participant_ids),
        "firstMessageAt": first_message_at.isoformat() if first_message_at else None,
        "lastMessageAt": last_message_at.isoformat() if last_message_at else None,
    }
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    ]


async def handle_tag_forum_post(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    normalize_name = deps["normalize_name"]

    server_id = arguments.get("server_id") or arguments.get("server")
    channel_identifier = arguments["channel"]
    post_id = _resolve_post_id(arguments)
    tag_names = _resolve_tag_names(arguments)

    forum_channel = await gateway.resolve_forum_channel(channel_identifier, server_id)
    thread, _ = await gateway.resolve_forum_post(post_id, server_id)
    if thread.parent_id != forum_channel.id:
        raise ValueError(
            f"Forum post '{thread.id}' does not belong to forum channel '{forum_channel.name}'"
        )

    selected_tags = _lookup_tags(forum_channel, normalize_name, tag_names)
    merged = {tag.id: tag for tag in thread.applied_tags}
    for tag in selected_tags:
        merged[tag.id] = tag
    next_tags = list(merged.values())

    await thread.edit(applied_tags=next_tags)
    return [
        TextContent(
            type="text",
            text=f"Applied tags to forum post '{thread.name}'. Applied tags: {', '.join(tag.name for tag in next_tags)}",
        )
    ]


async def handle_retag_forum_post(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    normalize_name = deps["normalize_name"]

    server_id = arguments.get("server_id") or arguments.get("server")
    channel_identifier = arguments["channel"]
    post_id = _resolve_post_id(arguments)
    tag_names = _resolve_tag_names(arguments)

    forum_channel = await gateway.resolve_forum_channel(channel_identifier, server_id)
    thread, _ = await gateway.resolve_forum_post(post_id, server_id)
    if thread.parent_id != forum_channel.id:
        raise ValueError(
            f"Forum post '{thread.id}' does not belong to forum channel '{forum_channel.name}'"
        )

    selected_tags = _lookup_tags(forum_channel, normalize_name, tag_names)
    await thread.edit(applied_tags=selected_tags)
    return [
        TextContent(
            type="text",
            text=f"Retagged forum post '{thread.name}' with tags: {', '.join(tag.name for tag in selected_tags)}",
        )
    ]
