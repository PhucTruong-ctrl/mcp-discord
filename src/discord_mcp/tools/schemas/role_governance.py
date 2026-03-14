from mcp.types import Tool


ROLE_GOVERNANCE_TOOLS = [
    Tool(
        name="create_role",
        description="Create a guild role",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "name": {"type": "string"},
                "permissions": {"type": "number"},
                "color": {"type": "number"},
                "hoist": {"type": "boolean"},
                "mentionable": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="delete_role",
        description="Delete a guild role",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "role_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "role_id"],
        },
    ),
    Tool(
        name="update_role",
        description="Update role properties",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "role_id": {"type": "string"},
                "name": {"type": "string"},
                "permissions": {"type": "number"},
                "color": {"type": "number"},
                "hoist": {"type": "boolean"},
                "mentionable": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "role_id"],
        },
    ),
    Tool(
        name="add_roles_bulk",
        description="Add multiple roles to multiple members",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "user_ids": {"type": "array", "items": {"type": "string"}},
                "role_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "user_ids", "role_ids"],
        },
    ),
    Tool(
        name="remove_roles_bulk",
        description="Remove multiple roles from multiple members",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "user_ids": {"type": "array", "items": {"type": "string"}},
                "role_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "user_ids", "role_ids"],
        },
    ),
    Tool(
        name="mute_member_role_based",
        description="Mute member by assigning mute role",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "user_id": {"type": "string"},
                "mute_role_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "user_id", "mute_role_id"],
        },
    ),
    Tool(
        name="unmute_member_role_based",
        description="Unmute member by removing mute role",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "user_id": {"type": "string"},
                "mute_role_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "user_id", "mute_role_id"],
        },
    ),
    Tool(
        name="permission_drift_check",
        description="Compare current role permissions with baseline",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "baseline_snapshot": {"type": "object"},
            },
            "required": ["server_id"],
        },
    ),
]
