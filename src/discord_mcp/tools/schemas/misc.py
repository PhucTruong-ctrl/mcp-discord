from mcp.types import Tool


MISC_TOOLS = [
    Tool(
        name="download_attachment",
        description="Download a Discord attachment URL to local filesystem",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Discord attachment URL",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename override",
                },
                "directory": {
                    "type": "string",
                    "description": "Output directory (default: current working directory)",
                },
            },
            "required": ["url"],
        },
    ),
    Tool(
        name="get_user_info",
        description="Get information about a Discord user",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "Discord user ID"}
            },
            "required": ["user_id"],
        },
    ),
    Tool(
        name="moderate_message",
        description="Delete a message and optionally timeout the user",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Channel ID containing the message",
                },
                "message_id": {
                    "type": "string",
                    "description": "ID of message to moderate",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for moderation",
                },
                "timeout_minutes": {
                    "type": "number",
                    "description": "Optional timeout duration in minutes",
                    "minimum": 0,
                    "maximum": 40320,
                },
            },
            "required": ["channel_id", "message_id", "reason"],
        },
    ),
]
