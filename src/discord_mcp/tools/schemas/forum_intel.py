from mcp.types import Tool


FORUM_INTEL_TOOLS = [
    Tool(
        name="list_forum_posts",
        description="List forum posts in a forum channel",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID (optional when default guild is configured)",
                },
                "channel": {
                    "type": "string",
                    "description": "Forum channel name or ID",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of posts to return",
                    "minimum": 1,
                    "maximum": 100,
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived posts",
                },
            },
            "required": ["channel"],
        },
    ),
    Tool(
        name="read_forum_post_messages",
        description="Read messages from a single forum post",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID (optional when default guild is configured)",
                },
                "post_id": {
                    "type": "string",
                    "description": "Forum post (thread) ID",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of messages to return",
                    "minimum": 1,
                    "maximum": 100,
                },
                "before": {
                    "type": "string",
                    "description": "Optional message ID for pagination",
                },
            },
            "required": ["post_id"],
        },
    ),
    Tool(
        name="read_forum_posts_batch",
        description="Read messages from multiple forum posts",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID (optional when default guild is configured)",
                },
                "post_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Forum post (thread) IDs",
                },
                "limit_per_post": {
                    "type": "number",
                    "description": "Maximum number of messages per post",
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["post_ids"],
        },
    ),
    Tool(
        name="get_thread_context",
        description="Get context for a forum post",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID (optional when default guild is configured)",
                },
                "post_id": {
                    "type": "string",
                    "description": "Forum post (thread) ID",
                },
            },
            "required": ["post_id"],
        },
    ),
    Tool(
        name="list_thread_participants",
        description="List participants in a forum post",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID (optional when default guild is configured)",
                },
                "post_id": {
                    "type": "string",
                    "description": "Forum post (thread) ID",
                },
            },
            "required": ["post_id"],
        },
    ),
    Tool(
        name="get_thread_activity_summary",
        description="Get activity summary for a forum post",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID (optional when default guild is configured)",
                },
                "post_id": {
                    "type": "string",
                    "description": "Forum post (thread) ID",
                },
            },
            "required": ["post_id"],
        },
    ),
    Tool(
        name="tag_forum_post",
        description="Add tags to a forum post",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID (optional when default guild is configured)",
                },
                "channel": {
                    "type": "string",
                    "description": "Forum channel name or ID",
                },
                "post_id": {
                    "type": "string",
                    "description": "Forum post (thread) ID",
                },
                "tag_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tag names to apply",
                },
            },
            "required": ["channel", "post_id", "tag_names"],
        },
    ),
    Tool(
        name="retag_forum_post",
        description="Replace tags on a forum post",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID (optional when default guild is configured)",
                },
                "channel": {
                    "type": "string",
                    "description": "Forum channel name or ID",
                },
                "post_id": {
                    "type": "string",
                    "description": "Forum post (thread) ID",
                },
                "tag_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tag names to set",
                },
            },
            "required": ["channel", "post_id", "tag_names"],
        },
    ),
]
