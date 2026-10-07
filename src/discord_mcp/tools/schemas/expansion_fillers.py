from mcp.types import Tool


EXPANSION_FILLER_TOOLS = [
    Tool(
        name="remove_member_timeout",
        description="Remove an active timeout from a member (member.timeout(None))",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "member_id": {"type": "string"},
                "reason": {"type": "string", "description": "Audit log reason"},
            },
            "required": ["server_id", "member_id"],
        },
    ),
    Tool(
        name="unban_member",
        description=(
            "Unban a user from the server (fails if the user is not banned). "
            "Returns the unbanned user."
        ),
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
        description=(
            "Ban up to 200 users in one request (guild.bulk_ban) with a dry_run/"
            "confirm_token gate. Requires ban_members."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "member_ids": {"type": "array", "items": {"type": "string"}},
                "delete_message_days": {"type": "integer", "minimum": 0, "maximum": 7},
                "dry_run": {"type": "boolean"},
                "confirm_token": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "member_ids"],
        },
    ),
    Tool(
        name="prune_inactive_members",
        description=(
            "Prune members inactive for N days (guild.prune_members) with a dry_run/"
            "confirm_token gate. Returns the pruned count."
        ),
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
        description="Create a channel category on the server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "name": {"type": "string"},
                "position": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="rename_category",
        description="Rename a category (rejects non-category channels)",
        inputSchema={
            "type": "object",
            "properties": {
                "category_id": {"type": "string"},
                "name": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["category_id", "name"],
        },
    ),
    Tool(
        name="move_category",
        description="Move a category to a new position in the channel list",
        inputSchema={
            "type": "object",
            "properties": {
                "category_id": {"type": "string"},
                "position": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["category_id", "position"],
        },
    ),
    Tool(
        name="delete_category",
        description=(
            "Delete a category with a dry_run/confirm_token gate. Child channels are "
            "kept and reported as orphaned."
        ),
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
        description=(
            "Create an incident channel (optionally inside a category), record its state "
            "and post an opening message."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "name": {"type": "string"},
                "reason": {"type": "string"},
                "category_id": {
                    "type": "string",
                    "description": "Optional parent category",
                },
            },
            "required": ["server_id", "name", "reason"],
        },
    ),
    Tool(
        name="append_incident_event",
        description=(
            "Post a timestamped event line to the incident channel and append it to the "
            "stored incident timeline (severity: low|medium|high|critical)."
        ),
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
        description=(
            "Close an incident: post the summary, deny @everyone send_messages on the "
            "channel and record the closing state (restorable via incident_rollback)."
        ),
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
        description="Create auto moderation rule with audit reason",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "rule": {"type": "object"},
                "reason": {
                    "type": "string",
                    "description": "Audit reason for the creation",
                },
            },
            "required": ["server_id", "rule"],
        },
    ),
    Tool(
        name="update_auto_moderation_rule",
        description="Update auto moderation rule with audit reason",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "rule_id": {"type": "string"},
                "rule": {"type": "object"},
                "reason": {
                    "type": "string",
                    "description": "Audit reason for the update",
                },
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
