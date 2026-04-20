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
            },
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="create_voice_channel",
        description="Create a new voice channel",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "name": {"type": "string", "description": "Channel name"},
                "bitrate": {
                    "type": "number",
                    "description": "Optional voice bitrate",
                },
                "user_limit": {
                    "type": "number",
                    "description": "Optional user limit",
                },
                "nsfw": {
                    "type": "boolean",
                    "description": "Mark channel as age-restricted",
                },
                "category_id": {
                    "type": "string",
                    "description": "Optional category ID to place channel in",
                },
                "position": {
                    "type": "number",
                    "description": "Optional channel position",
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
                "topic": {
                    "type": "string",
                    "description": "Optional channel topic or guidelines",
                },
                "available_tags": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional available forum tags",
                },
                "default_auto_archive_duration": {
                    "type": "number",
                    "description": "Optional default auto-archive duration",
                },
                "slowmode_delay": {
                    "type": "number",
                    "description": "Optional slowmode delay",
                },
                "category_id": {
                    "type": "string",
                    "description": "Optional category ID to place channel in",
                },
                "position": {
                    "type": "number",
                    "description": "Optional channel position",
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
                "channel_id": {"type": "string", "description": "Channel ID"},
                "name": {"type": "string", "description": "Channel name"},
                "topic": {"type": "string", "description": "Channel topic"},
                "nsfw": {
                    "type": "boolean",
                    "description": "Mark channel as age-restricted",
                },
                "slowmode_delay": {
                    "type": "number",
                    "description": "Slowmode delay in seconds",
                },
                "category_id": {
                    "type": "string",
                    "description": "Category ID to move channel into",
                },
                "position": {"type": "number", "description": "Channel position"},
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
                "channel_id": {"type": "string", "description": "Channel ID"},
                "name": {"type": "string", "description": "Channel name"},
                "bitrate": {"type": "number", "description": "Voice bitrate"},
                "user_limit": {"type": "number", "description": "User limit"},
                "nsfw": {
                    "type": "boolean",
                    "description": "Mark channel as age-restricted",
                },
                "category_id": {
                    "type": "string",
                    "description": "Category ID to move channel into",
                },
                "position": {"type": "number", "description": "Channel position"},
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
                "channel_id": {"type": "string", "description": "Channel ID"},
                "name": {"type": "string", "description": "Channel name"},
                "topic": {
                    "type": "string",
                    "description": "Channel topic or guidelines",
                },
                "available_tags": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Available forum tags",
                },
                "default_auto_archive_duration": {
                    "type": "number",
                    "description": "Default auto-archive duration",
                },
                "slowmode_delay": {
                    "type": "number",
                    "description": "Slowmode delay in seconds",
                },
                "category_id": {
                    "type": "string",
                    "description": "Category ID to move channel into",
                },
                "position": {"type": "number", "description": "Channel position"},
            },
            "required": ["server_id", "channel_id"],
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
            },
            "required": ["channel_id"],
        },
    ),
]
