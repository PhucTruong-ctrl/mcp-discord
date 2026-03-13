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
