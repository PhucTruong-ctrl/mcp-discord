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
]
