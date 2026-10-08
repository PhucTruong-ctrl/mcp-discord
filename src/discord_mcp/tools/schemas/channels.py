from mcp.types import Tool


CHANNEL_TOOLS = [
    Tool(
        name="create_text_channel",
        description="Create a new text channel",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "name": {"type": "string", "description": "Channel name"},
                "category_id": {
                    "type": "string",
                    "description": "Optional category ID to place channel in",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional channel topic",
                },
                "position": {"type": "number", "description": "Optional channel position"},
                "nsfw": {"type": "boolean", "description": "Mark channel as NSFW"},
                "slowmode_delay": {
                    "type": "number",
                    "description": "Optional slowmode delay in seconds",
                },
                "default_auto_archive_duration": {
                    "type": "number",
                    "description": "Optional default auto archive duration (minutes)",
                },
                "default_thread_slowmode_delay": {
                    "type": "number",
                    "description": "Optional default thread slowmode delay",
                },
                "reason": {"type": "string", "description": "Reason for creation"},
            },
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="delete_channel",
        description="Delete a channel",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "ID of channel to delete",
                },
                "reason": {"type": "string", "description": "Reason for deletion"},
                "dry_run": {"type": "boolean"},
                "confirm_token": {"type": "string"},
            },
            "required": ["channel_id", "reason"],
        },
    ),
]


CHANNEL_ADMIN_TOOLS = [
    Tool(
        name="create_voice_channel",
        description="Create a new voice channel",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "name": {"type": "string", "description": "Channel name"},
                "category_id": {
                    "type": "string",
                    "description": "Optional category ID to place channel in",
                },
                "bitrate": {"type": "number", "description": "Optional bitrate"},
                "user_limit": {
                    "type": "number",
                    "description": "Optional user limit",
                },
                "rtc_region": {
                    "type": "string",
                    "description": "Optional RTC region",
                },
                "reason": {"type": "string", "description": "Reason for creation"},
                "video_quality_mode": {
                    "type": "number",
                    "description": "Optional video quality mode",
                },
            },
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="create_forum_channel",
        description="Create a new forum channel",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "name": {"type": "string", "description": "Channel name"},
                "category_id": {
                    "type": "string",
                    "description": "Optional category ID to place channel in",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional channel topic",
                },
                "nsfw": {"type": "boolean", "description": "Mark channel as NSFW"},
                "slowmode_delay": {
                    "type": "number",
                    "description": "Optional slowmode delay",
                },
                "default_auto_archive_duration": {
                    "type": "number",
                    "description": "Optional default auto archive duration",
                },
                "default_reaction_emoji": {
                    "type": "object",
                    "description": "Optional default reaction emoji",
                },
                "default_layout": {
                    "type": "number",
                    "description": "Optional default forum layout: 0 not_set, 1 list_view, 2 gallery_view",
                },
                "default_sort_order": {
                    "type": "number",
                    "description": "Optional default forum sort order: 0 latest_activity, 1 creation_date",
                },
                "default_thread_slowmode_delay": {
                    "type": "number",
                    "description": "Optional default thread slowmode delay",
                },
                "position": {"type": "number", "description": "Optional channel position"},
                "reason": {"type": "string", "description": "Reason for creation"},
                "available_tags": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional available forum tags",
                },
            },
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="update_text_channel",
        description="Update an existing text channel",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {
                    "type": "string",
                    "description": "ID of text channel to update",
                },
                "name": {"type": "string", "description": "Channel name"},
                "category_id": {
                    "type": "string",
                    "description": "Optional category ID to move channel into",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional channel topic",
                },
                "nsfw": {"type": "boolean", "description": "Mark channel as NSFW"},
                "slowmode_delay": {
                    "type": "number",
                    "description": "Optional slowmode delay",
                },
                "position": {
                    "type": "number",
                    "description": "Optional channel position",
                },
                "reason": {"type": "string", "description": "Reason for update"},
            },
            "required": ["server_id", "channel_id"],
        },
    ),
    Tool(
        name="update_voice_channel",
        description="Update an existing voice channel",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {
                    "type": "string",
                    "description": "ID of voice channel to update",
                },
                "name": {"type": "string", "description": "Channel name"},
                "category_id": {
                    "type": "string",
                    "description": "Optional category ID to move channel into",
                },
                "bitrate": {"type": "number", "description": "Optional bitrate"},
                "user_limit": {
                    "type": "number",
                    "description": "Optional user limit",
                },
                "rtc_region": {
                    "type": "string",
                    "description": "Optional RTC region",
                },
                "video_quality_mode": {
                    "type": "number",
                    "description": "Optional video quality mode",
                },
                "position": {
                    "type": "number",
                    "description": "Optional channel position",
                },
                "reason": {"type": "string", "description": "Reason for update"},
            },
            "required": ["server_id", "channel_id"],
        },
    ),
    Tool(
        name="update_forum_channel",
        description="Update an existing forum channel",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {
                    "type": "string",
                    "description": "ID of forum channel to update",
                },
                "name": {"type": "string", "description": "Channel name"},
                "category_id": {
                    "type": "string",
                    "description": "Optional category ID to move channel into",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional channel topic",
                },
                "nsfw": {"type": "boolean", "description": "Mark channel as NSFW"},
                "slowmode_delay": {
                    "type": "number",
                    "description": "Optional slowmode delay",
                },
                "default_auto_archive_duration": {
                    "type": "number",
                    "description": "Optional default auto archive duration",
                },
                "default_reaction_emoji": {
                    "type": "object",
                    "description": "Optional default reaction emoji",
                },
                "available_tags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Existing tag id. Required to keep a tag: Discord "
                                "keys applied tags by id, so a rewrite without ids detaches the "
                                "tag from every post that had it.",
                            },
                            "name": {"type": "string", "description": "Tag name (max 20 chars)"},
                            "emoji": {"type": "string", "description": "Tag emoji"},
                            "emojiId": {
                                "type": ["string", "null"],
                                "description": "Custom emoji id; leave null for a standard emoji",
                            },
                            "emojiAnimated": {"type": "boolean", "description": "Animated emoji"},
                            "moderated": {"type": "boolean", "description": "Can only moderators apply"},
                        },
                        "required": ["name"],
                    },
                    "description": "Replaces the forum's tag list. Read the current tags with "
                    "get_channels_structured and send them back with their ids to rename safely.",
                },
                "allow_recreate_tags": {
                    "type": "boolean",
                    "description": "Set true to allow replacing the tag list without ids. Every "
                    "forum post tagged with a dropped tag loses it, and this is not recoverable "
                    "through the API. Defaults to false.",
                },
                "position": {
                    "type": "number",
                    "description": "Optional channel position",
                },
                "default_layout": {
                    "type": "number",
                    "description": "Optional default forum layout: 0 not_set, 1 list_view, 2 gallery_view",
                },
                "default_sort_order": {
                    "type": "number",
                    "description": "Optional default forum sort order: 0 latest_activity, 1 creation_date",
                },
                "default_thread_slowmode_delay": {
                    "type": "number",
                    "description": "Optional default thread slowmode delay",
                },
                "reason": {"type": "string", "description": "Reason for update"},
            },
            "required": ["server_id", "channel_id"],
        },
    ),
]
