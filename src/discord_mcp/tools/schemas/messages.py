from mcp.types import Tool


MESSAGE_TOOLS = [
    Tool(
        name="add_reaction",
        description="Add a reaction to a message",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID",
                },
                "channel_id": {
                    "type": "string",
                    "description": "Channel containing the message",
                },
                "message_id": {
                    "type": "string",
                    "description": "Message to react to",
                },
                "emoji": {
                    "type": "string",
                    "description": "Emoji to react with (Unicode or custom emoji ID)",
                },
            },
            "required": ["channel_id", "message_id", "emoji"],
        },
    ),
    Tool(
        name="add_multiple_reactions",
        description="Add multiple reactions to a message",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID",
                },
                "channel_id": {
                    "type": "string",
                    "description": "Channel containing the message",
                },
                "message_id": {
                    "type": "string",
                    "description": "Message to react to",
                },
                "emojis": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Emoji to react with (Unicode or custom emoji ID)",
                    },
                    "description": "List of emojis to add as reactions",
                },
            },
            "required": ["channel_id", "message_id", "emojis"],
        },
    ),
    Tool(
        name="remove_reaction",
        description="Remove a reaction from a message",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server ID",
                },
                "channel_id": {
                    "type": "string",
                    "description": "Channel containing the message",
                },
                "message_id": {
                    "type": "string",
                    "description": "Message to remove reaction from",
                },
                "emoji": {
                    "type": "string",
                    "description": "Emoji to remove (Unicode or custom emoji ID)",
                },
            },
            "required": ["channel_id", "message_id", "emoji"],
        },
    ),
    Tool(
        name="send_message",
        description="Send a message to a specific channel",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Discord channel ID",
                },
                "content": {"type": "string", "description": "Message content"},
            },
            "required": ["channel_id", "content"],
        },
    ),
    Tool(
        name="read_messages",
        description="Read recent messages from a channel",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Discord channel ID",
                },
                "limit": {
                    "type": "number",
                    "description": "Number of messages to fetch (max 100)",
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["channel_id"],
        },
    ),
    Tool(
        name="edit_message",
        description="Edit a message sent by the bot",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Discord channel ID",
                },
                "message_id": {
                    "type": "string",
                    "description": "Message ID to edit",
                },
                "content": {
                    "type": "string",
                    "description": "New message content",
                },
            },
            "required": ["channel_id", "message_id", "content"],
        },
    ),
]
