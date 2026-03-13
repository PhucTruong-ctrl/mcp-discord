from mcp.types import Tool


FORUM_TOOLS = [
    Tool(
        name="read_forum_threads",
        description="Read active forum threads and recent posts",
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
                    "description": "Number of threads to fetch (max 50)",
                    "minimum": 1,
                    "maximum": 50,
                },
                "before": {
                    "type": "string",
                    "description": "Optional message ID for history pagination",
                },
            },
            "required": ["channel"],
        },
    ),
    Tool(
        name="list_threads",
        description="List forum threads without reading full message history",
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
                    "description": "Maximum number of threads to return",
                    "minimum": 1,
                    "maximum": 100,
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived threads",
                },
            },
            "required": ["channel"],
        },
    ),
    Tool(
        name="search_threads",
        description="Search forum threads by title",
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
                "query": {
                    "type": "string",
                    "description": "Thread title query",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of threads to return",
                    "minimum": 1,
                    "maximum": 100,
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived threads in search",
                },
                "exact_match": {
                    "type": "boolean",
                    "description": "Require exact thread title match",
                },
            },
            "required": ["channel", "query"],
        },
    ),
    Tool(
        name="add_thread_tags",
        description="Add tags to a forum thread",
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
                "thread_id": {
                    "type": "string",
                    "description": "Thread ID",
                },
                "tag_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of tag names to apply",
                },
            },
            "required": ["channel", "thread_id", "tag_names"],
        },
    ),
    Tool(
        name="unarchive_thread",
        description="Unarchive (reopen) a forum thread",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID (optional when default guild is configured)",
                },
                "thread_id": {
                    "type": "string",
                    "description": "Thread ID",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional reason",
                },
            },
            "required": ["thread_id"],
        },
    ),
]
