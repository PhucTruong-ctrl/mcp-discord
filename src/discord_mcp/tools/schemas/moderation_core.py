from mcp.types import Tool


MODERATION_CORE_TOOLS = [
    Tool(
        name="moderation_bulk_delete",
        description="Bulk delete messages with dry-run and confirm token enforcement",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Target channel ID"},
                "message_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Message IDs to delete",
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {"type": "boolean", "description": "Return dry-run result"},
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["channel_id", "message_ids"],
        },
    ),
    Tool(
        name="moderation_timeout_member",
        description="Timeout a member with dry-run and confirm token enforcement",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member ID"},
                "duration_minutes": {
                    "type": "number",
                    "description": "Timeout duration in minutes",
                    "minimum": 1,
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {"type": "boolean", "description": "Return dry-run result"},
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "member_id", "duration_minutes"],
        },
    ),
    Tool(
        name="moderation_kick_member",
        description="Kick a member with dry-run and confirm token enforcement",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member ID"},
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {"type": "boolean", "description": "Return dry-run result"},
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "member_id"],
        },
    ),
    Tool(
        name="moderation_ban_member",
        description="Ban a member with dry-run and confirm token enforcement",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member ID"},
                "delete_message_days": {
                    "type": "number",
                    "description": "Delete message history days",
                    "minimum": 0,
                    "maximum": 7,
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {"type": "boolean", "description": "Return dry-run result"},
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "member_id"],
        },
    ),
]
