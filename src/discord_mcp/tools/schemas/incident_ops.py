from mcp.types import Tool


INCIDENT_OPS_TOOLS = [
    Tool(
        name="incident_get_channel_state",
        description=(
            "Read the stored incident state for a channel (lockdown snapshot, closed flag, "
            "logged events) from the MCP state store."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Discord channel ID"}
            },
            "required": ["channel_id"],
        },
    ),
    Tool(
        name="incident_set_channel_state",
        description=(
            "Overwrite the stored incident state for a channel (persisted to the MCP "
            "state file so it survives restarts)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Discord channel ID"},
                "state": {
                    "type": "object",
                    "description": "State model containing permission and slowdown controls",
                },
                "reason": {
                    "type": "string",
                    "description": "Audit reason for the state overwrite",
                },
                "dry_run": {"type": "boolean"},
                "confirm_token": {"type": "string"},
            },
            "required": ["channel_id", "state", "reason"],
        },
    ),
    Tool(
        name="incident_apply_lockdown",
        description=(
            "Lock a channel down: snapshot the current @everyone overwrite, then deny "
            "send_messages / send_messages_in_threads / create_public_threads for "
            "@everyone. dry_run by default; confirm_token required to apply."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "channel_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target channel IDs",
                },
                "reason": {"type": "string", "description": "Required audit reason"},
                "dry_run": {
                    "type": "boolean",
                    "description": "Return confirm token without applying when true",
                    "default": True,
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Required when dry_run is false",
                },
            },
            "required": ["channel_ids", "reason"],
        },
    ),
    Tool(
        name="incident_rollback_lockdown",
        description=(
            "Undo a lockdown by restoring the @everyone overwrite snapshot recorded when it "
            "was applied (or removing the overwrite if none existed)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "channel_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Target channel IDs",
                },
                "reason": {"type": "string", "description": "Required audit reason"},
                "dry_run": {
                    "type": "boolean",
                    "description": "Return confirm token without rollback when true",
                    "default": True,
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Required when dry_run is false",
                },
            },
            "required": ["channel_ids", "reason"],
        },
    ),
]
