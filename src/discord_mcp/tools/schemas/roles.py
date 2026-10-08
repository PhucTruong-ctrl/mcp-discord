from mcp.types import Tool


ROLE_TOOLS = [
    Tool(
        name="add_role",
        description="Add a role to a user (dry-run by default)",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "user_id": {"type": "string", "description": "User to add role to"},
                "role_id": {"type": "string", "description": "Role ID to add"},
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return dry-run result",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "user_id", "role_id", "reason"],
        },
    ),
    Tool(
        name="remove_role",
        description="Remove a role from a user (dry-run by default)",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "user_id": {
                    "type": "string",
                    "description": "User to remove role from",
                },
                "role_id": {"type": "string", "description": "Role ID to remove"},
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return dry-run result",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "user_id", "role_id", "reason"],
        },
    ),
]
