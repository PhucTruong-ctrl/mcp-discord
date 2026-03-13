import json
from typing import Any, Dict, List

import discord
from mcp.types import TextContent


async def handle_read_forum_threads(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    try_int = deps["try_int"]
    serialize_message = deps["serialize_message"]
    serialize_forum_tag = deps["serialize_forum_tag"]

    server_id = arguments.get("server_id") or arguments.get("server")
    channel_identifier = arguments["channel"]
    limit = min(int(arguments.get("limit", 10)), 50)
    before = arguments.get("before")
    before_obj = discord.Object(id=int(before)) if try_int(before) else None

    forum_channel = await gateway.resolve_forum_channel(channel_identifier, server_id)
    threads = sorted(
        list(forum_channel.threads),
        key=lambda t: t.created_at or discord.utils.utcnow(),
        reverse=True,
    )[:limit]

    thread_payload = []
    for thread in threads:
        messages = []
        async for msg in thread.history(limit=10, before=before_obj):
            payload = serialize_message(msg)
            payload.update(
                {
                    "thread": thread.name,
                    "threadId": str(thread.id),
                    "channel": f"#{forum_channel.name}",
                    "server": thread.guild.name,
                }
            )
            messages.append(payload)

        thread_payload.append(
            {
                "thread": thread.name,
                "threadId": str(thread.id),
                "tags": [serialize_forum_tag(tag) for tag in thread.applied_tags],
                "messages": messages,
            }
        )

    return [
        TextContent(
            type="text", text=json.dumps(thread_payload, ensure_ascii=False, indent=2)
        )
    ]


async def handle_list_threads(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    serialize_forum_tag = deps["serialize_forum_tag"]
    max_archived_threads_scan = deps["max_archived_threads_scan"]

    server_id = arguments.get("server_id") or arguments.get("server")
    channel_identifier = arguments["channel"]
    limit = min(int(arguments.get("limit", 50)), 100)
    include_archived = bool(
        arguments.get("include_archived", arguments.get("includeArchived", False))
    )

    forum_channel = await gateway.resolve_forum_channel(channel_identifier, server_id)
    all_threads = await gateway.collect_forum_threads(
        forum_channel, include_archived, max_archived_threads_scan
    )
    thread_list = sorted(
        all_threads, key=lambda t: t.created_at or discord.utils.utcnow(), reverse=True
    )[:limit]

    result = {
        "forumChannel": forum_channel.name,
        "server": forum_channel.guild.name,
        "includeArchived": include_archived,
        "totalThreads": len(thread_list),
        "threads": [
            {
                "threadId": str(thread.id),
                "threadName": thread.name,
                "createdAt": thread.created_at.isoformat()
                if thread.created_at
                else None,
                "ownerId": str(thread.owner_id) if thread.owner_id else None,
                "archived": thread.archived,
                "locked": thread.locked,
                "messageCount": thread.message_count,
                "memberCount": thread.member_count,
                "totalMessageSent": thread.total_message_sent,
                "rateLimitPerUser": thread.slowmode_delay,
                "tags": [serialize_forum_tag(tag) for tag in thread.applied_tags],
                "lastMessageId": str(thread.last_message_id)
                if thread.last_message_id
                else None,
            }
            for thread in thread_list
        ],
    }

    return [
        TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))
    ]


async def handle_search_threads(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    serialize_forum_tag = deps["serialize_forum_tag"]
    max_archived_threads_scan = deps["max_archived_threads_scan"]

    server_id = arguments.get("server_id") or arguments.get("server")
    channel_identifier = arguments["channel"]
    query = str(arguments["query"]).strip()
    limit = min(int(arguments.get("limit", 50)), 100)
    include_archived = bool(
        arguments.get("include_archived", arguments.get("includeArchived", True))
    )
    exact_match = bool(arguments.get("exact_match", arguments.get("exactMatch", False)))

    forum_channel = await gateway.resolve_forum_channel(channel_identifier, server_id)
    all_threads = await gateway.collect_forum_threads(
        forum_channel, include_archived, max_archived_threads_scan
    )
    query_lower = query.lower()

    def matches(thread: discord.Thread) -> bool:
        name_lower = thread.name.lower()
        if exact_match:
            return name_lower == query_lower
        return query_lower in name_lower

    filtered = [thread for thread in all_threads if matches(thread)]
    thread_list = sorted(
        filtered, key=lambda t: t.created_at or discord.utils.utcnow(), reverse=True
    )[:limit]

    result = {
        "forumChannel": forum_channel.name,
        "server": forum_channel.guild.name,
        "query": query,
        "exactMatch": exact_match,
        "totalFound": len(thread_list),
        "totalMatched": len(filtered),
        "includeArchived": include_archived,
        "threads": [
            {
                "threadId": str(thread.id),
                "threadName": thread.name,
                "createdAt": thread.created_at.isoformat()
                if thread.created_at
                else None,
                "ownerId": str(thread.owner_id) if thread.owner_id else None,
                "archived": thread.archived,
                "locked": thread.locked,
                "messageCount": thread.message_count,
                "memberCount": thread.member_count,
                "totalMessageSent": thread.total_message_sent,
                "rateLimitPerUser": thread.slowmode_delay,
                "tags": [serialize_forum_tag(tag) for tag in thread.applied_tags],
                "lastMessageId": str(thread.last_message_id)
                if thread.last_message_id
                else None,
            }
            for thread in thread_list
        ],
    }

    return [
        TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))
    ]


async def handle_add_thread_tags(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    normalize_name = deps["normalize_name"]

    server_id = arguments.get("server_id") or arguments.get("server")
    channel_identifier = arguments["channel"]
    thread_id = arguments.get("thread_id") or arguments.get("threadId")
    tag_names = arguments.get("tag_names") or arguments.get("tagNames")

    if not thread_id:
        raise ValueError("thread_id is required")
    if not isinstance(tag_names, list) or not tag_names:
        raise ValueError("tag_names must be a non-empty array")

    forum_channel = await gateway.resolve_forum_channel(channel_identifier, server_id)
    thread, _ = await gateway.resolve_thread(str(thread_id), server_id)

    if thread.parent_id != forum_channel.id:
        raise ValueError(
            f"Thread '{thread.id}' does not belong to forum channel '{forum_channel.name}'"
        )

    requested = {normalize_name(tag): tag for tag in tag_names}
    tag_lookup = {normalize_name(tag.name): tag for tag in forum_channel.available_tags}

    missing = [original for key, original in requested.items() if key not in tag_lookup]
    if missing:
        available = ", ".join(tag.name for tag in forum_channel.available_tags)
        raise ValueError(
            f"Tags not found: {', '.join(missing)}. Available tags: {available}"
        )

    merged = {tag.id: tag for tag in thread.applied_tags}
    for key in requested:
        tag = tag_lookup[key]
        merged[tag.id] = tag

    await thread.edit(applied_tags=list(merged.values()))
    applied_names = [tag.name for tag in merged.values()]
    return [
        TextContent(
            type="text",
            text=f"Tags added to thread '{thread.name}'. Applied tags: {', '.join(applied_names)}",
        )
    ]


async def handle_unarchive_thread(
    arguments: Dict[str, Any], deps: Dict[str, Any]
) -> List[TextContent]:
    gateway = deps["gateway"]
    server_id = arguments.get("server_id") or arguments.get("server")
    thread_id = arguments.get("thread_id") or arguments.get("threadId")
    reason = arguments.get("reason")

    if not thread_id:
        raise ValueError("thread_id is required")

    thread, _ = await gateway.resolve_thread(str(thread_id), server_id)
    if not thread.archived:
        return [
            TextContent(type="text", text=f"Thread '{thread.name}' is already active.")
        ]

    await thread.edit(archived=False, reason=reason)
    return [
        TextContent(type="text", text=f"Thread '{thread.name}' has been unarchived.")
    ]
