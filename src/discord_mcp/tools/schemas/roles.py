from mcp.types import Tool


ROLE_TOOLS = [
    Tool(
        name="add_role",
        description="Add a role to a user",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "user_id": {"type": "string", "description": "User to add role to"},
                "role_id": {"type": "string", "description": "Role ID to add"},
            },
            "required": ["server_id", "user_id", "role_id"],
        },
    ),
    Tool(
        name="remove_role",
        description="Remove a role from a user",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "user_id": {
                    "type": "string",
                    "description": "User to remove role from",
                },
                "role_id": {"type": "string", "description": "Role ID to remove"},
            },
            "required": ["server_id", "user_id", "role_id"],
        },
    ),
]
