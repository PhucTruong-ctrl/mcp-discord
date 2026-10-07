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
                "color": {
                    "type": ["number", "string"],
                    "description": "Primary colour: int, '#rrggbb' or '0xrrggbb'",
                },
                "secondary_color": {
                    "type": ["number", "string"],
                    "description": "Gradient colour 2 (int or hex)",
                },
                "tertiary_color": {
                    "type": ["number", "string"],
                    "description": "Gradient colour 3 (int or hex)",
                },
                "hoist": {"type": "boolean"},
                "mentionable": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="delete_role",
        description="Delete a guild role (by role_id or unique role_name)",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "role_id": {"type": "string"},
                "role_name": {
                    "type": "string",
                    "description": "Role name, case-insensitive; must match exactly one role",
                },
                "reason": {"type": "string"},
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="update_role",
        description=(
            "Update role properties (name, permissions, color, secondary/tertiary gradient colour, "
            "hoist, mentionable). Works on integration-managed roles too as long as the role sits "
            "below the bot's highest role; only the hierarchy blocks it."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "role_id": {
                    "type": "string",
                    "description": "Role ID (or use role_name)",
                },
                "role_name": {
                    "type": "string",
                    "description": "Role name, case-insensitive; must match exactly one role",
                },
                "name": {"type": "string", "description": "New role name"},
                "permissions": {"type": "number"},
                "color": {
                    "type": ["number", "string"],
                    "description": "Primary colour: int, '#rrggbb', '0xrrggbb' or null to clear",
                },
                "secondary_color": {
                    "type": ["number", "string", "null"],
                    "description": "Gradient colour 2 (int/hex) or null to clear the gradient stop",
                },
                "tertiary_color": {
                    "type": ["number", "string", "null"],
                    "description": "Gradient colour 3 (int/hex) or null to clear the gradient stop",
                },
                "hoist": {"type": "boolean"},
                "mentionable": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["server_id"],
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
                "dry_run": {"type": "boolean"},
                "confirm_token": {"type": "string"},
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
                "dry_run": {"type": "boolean"},
                "confirm_token": {"type": "string"},
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
        description=(
            "Compare current role permission bitfields with a baseline. Without "
            "baseline_snapshot the current state is returned as the baseline "
            "(mode=baseline); with one (the export_server_snapshot payload, or this "
            "tool's own baseline payload) it diffs every role and reports added/removed "
            "permission names (mode=drift)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "baseline_snapshot": {
                    "type": "object",
                    "description": (
                        "Prior snapshot: {'roles': [{'id', 'permissions', ...}]}. "
                        "Accepts export_server_snapshot output as-is."
                    ),
                },
            },
            "required": ["server_id"],
        },
    ),
]
