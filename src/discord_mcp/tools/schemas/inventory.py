from mcp.types import Tool


INVENTORY_TOOLS = [
    Tool(
        name="get_channels_structured",
        description="Return structured channel inventory for a server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"}
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="get_channel_hierarchy",
        description="Return category-channel hierarchy for a server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"}
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="get_role_hierarchy",
        description=(
            "Return roles sorted by hierarchy with permission bitfields and decoded "
            "permission names (supports the permission_drift_check baseline shape)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"}
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="get_permission_overwrites",
        description="Return explicit permission overwrites for a channel",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Channel ID to inspect",
                }
            },
            "required": ["channel_id"],
        },
    ),
    Tool(
        name="diff_channel_permissions",
        description="Diff permission overwrites between two channels",
        inputSchema={
            "type": "object",
            "properties": {
                "source_channel_id": {"type": "string"},
                "target_channel_id": {"type": "string"},
            },
            "required": ["source_channel_id", "target_channel_id"],
        },
    ),
    Tool(
        name="export_server_snapshot",
        description=(
            "Export a structural snapshot for a server (channels plus role permission "
            "bitfields, hoist and mentionable). The payload is a valid "
            "permission_drift_check baseline_snapshot."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"}
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="get_channel_type_counts",
        description="Count channels by Discord channel type",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"}
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="list_inactive_channels",
        description="List inactive text channels based on last message age",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "days": {
                    "type": "number",
                    "description": "Inactive threshold in days",
                    "minimum": 1,
                },
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="set_channel_permission_overwrite",
        description=(
            "Create or replace a channel permission overwrite for a role or member. "
            "allow/deny accept permission names (e.g. send_messages) or raw bit values; "
            "omitted allow/deny default to 0."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel ID"},
                "target_id": {
                    "type": "string",
                    "description": "Role or member ID",
                },
                "target_type": {
                    "type": "string",
                    "description": "role|member; auto-detected from cache when omitted",
                },
                "allow": {
                    "type": "array",
                    "items": {"type": ["string", "number"]},
                    "description": "Permission names or bit values to allow",
                },
                "deny": {
                    "type": "array",
                    "items": {"type": ["string", "number"]},
                    "description": "Permission names or bit values to deny",
                },
                "reason": {"type": "string", "description": "Audit log reason"},
            },
            "required": ["channel_id", "target_id"],
        },
    ),
    Tool(
        name="remove_channel_permission_overwrite",
        description=(
            "Delete a channel permission overwrite for a role or member "
            "(explicit overwrite only, inherited state is untouched; dry-run by default)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel ID"},
                "target_id": {
                    "type": "string",
                    "description": "Role or member ID",
                },
                "target_type": {
                    "type": "string",
                    "description": "role|member; auto-detected from cache when omitted",
                },
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
            "required": ["channel_id", "target_id", "reason"],
        },
    ),
]
