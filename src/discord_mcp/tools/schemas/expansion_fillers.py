from mcp.types import Tool


EXPANSION_FILLER_TOOLS = [
    Tool(
        name="remove_member_timeout",
        description="Remove timeout from a member",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "member_id": {"type": "string"},
            },
            "required": ["server_id", "member_id"],
        },
    ),
    Tool(
        name="unban_member",
        description="Unban a member",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "member_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "member_id"],
        },
    ),
    Tool(
        name="bulk_ban_members",
        description="Bulk ban members with confirmation gate",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "member_ids": {"type": "array", "items": {"type": "string"}},
                "dry_run": {"type": "boolean"},
                "confirm_token": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "member_ids"],
        },
    ),
    Tool(
        name="prune_inactive_members",
        description="Prune inactive members with confirmation gate",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "days": {"type": "integer"},
                "dry_run": {"type": "boolean"},
                "confirm_token": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "days"],
        },
    ),
    Tool(
        name="create_category",
        description="Create category",
        inputSchema={
            "type": "object",
            "properties": {"server_id": {"type": "string"}, "name": {"type": "string"}},
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="rename_category",
        description="Rename category",
        inputSchema={
            "type": "object",
            "properties": {
                "category_id": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["category_id", "name"],
        },
    ),
    Tool(
        name="move_category",
        description="Move category",
        inputSchema={
            "type": "object",
            "properties": {
                "category_id": {"type": "string"},
                "position": {"type": "integer"},
            },
            "required": ["category_id", "position"],
        },
    ),
    Tool(
        name="delete_category",
        description="Delete category with confirmation gate",
        inputSchema={
            "type": "object",
            "properties": {
                "category_id": {"type": "string"},
                "dry_run": {"type": "boolean"},
                "confirm_token": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["category_id"],
        },
    ),
    Tool(
        name="create_incident_room",
        description="Create incident room",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "name": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "name", "reason"],
        },
    ),
    Tool(
        name="append_incident_event",
        description="Append incident event",
        inputSchema={
            "type": "object",
            "properties": {
                "incident_channel_id": {"type": "string"},
                "event_text": {"type": "string"},
                "severity": {"type": "string"},
            },
            "required": ["incident_channel_id", "event_text", "severity"],
        },
    ),
    Tool(
        name="close_incident",
        description="Close incident",
        inputSchema={
            "type": "object",
            "properties": {
                "incident_channel_id": {"type": "string"},
                "summary": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["incident_channel_id", "summary", "reason"],
        },
    ),
    Tool(
        name="list_auto_moderation_rules",
        description="List auto moderation rules",
        inputSchema={
            "type": "object",
            "properties": {"server_id": {"type": "string"}},
            "required": ["server_id"],
        },
    ),
    Tool(
        name="create_auto_moderation_rule",
        description="Create auto moderation rule",
        inputSchema={
            "type": "object",
            "properties": {"server_id": {"type": "string"}, "rule": {"type": "object"}},
            "required": ["server_id", "rule"],
        },
    ),
    Tool(
        name="update_auto_moderation_rule",
        description="Update auto moderation rule",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "rule_id": {"type": "string"},
                "rule": {"type": "object"},
            },
            "required": ["server_id", "rule_id", "rule"],
        },
    ),
    Tool(
        name="automod_export_rules",
        description="Export automod rules",
        inputSchema={
            "type": "object",
            "properties": {"server_id": {"type": "string"}},
            "required": ["server_id"],
        },
    ),
]
