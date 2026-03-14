from mcp.types import Tool


MESSAGING_WORKFLOW_TOOLS = [
    Tool(
        name="send_embed_message",
        description="Send message content with an embed",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "channel_id": {"type": "string"},
                "content": {"type": "string"},
                "embed": {"type": "object"},
            },
            "required": ["channel_id", "embed"],
        },
    ),
    Tool(
        name="send_rich_announcement",
        description="Send an announcement payload with title/body fields",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "channel_id": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "color": {"type": "number"},
            },
            "required": ["channel_id", "title", "body"],
        },
    ),
    Tool(
        name="crosspost_announcement",
        description="Crosspost an announcement message in a news channel",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string"},
                "message_id": {"type": "string"},
            },
            "required": ["channel_id", "message_id"],
        },
    ),
    Tool(
        name="create_channel_webhook",
        description="Create webhook for a channel",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string"},
                "name": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["channel_id", "name"],
        },
    ),
    Tool(
        name="list_channel_webhooks",
        description="List webhooks in a channel",
        inputSchema={
            "type": "object",
            "properties": {"channel_id": {"type": "string"}},
            "required": ["channel_id"],
        },
    ),
    Tool(
        name="execute_channel_webhook",
        description="Execute channel webhook using caller-supplied id/token",
        inputSchema={
            "type": "object",
            "properties": {
                "webhook_id": {"type": "string"},
                "token": {"type": "string"},
                "content": {"type": "string"},
                "username": {"type": "string"},
            },
            "required": ["webhook_id", "token", "content"],
        },
    ),
    Tool(
        name="list_guild_integrations",
        description="List configured integrations in guild",
        inputSchema={
            "type": "object",
            "properties": {"server_id": {"type": "string"}},
            "required": ["server_id"],
        },
    ),
    Tool(
        name="get_guild_vanity_url",
        description="Get vanity URL for guild",
        inputSchema={
            "type": "object",
            "properties": {"server_id": {"type": "string"}},
            "required": ["server_id"],
        },
    ),
]
