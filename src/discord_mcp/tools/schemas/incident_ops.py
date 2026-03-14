from mcp.types import Tool


INCIDENT_OPS_TOOLS = [
    Tool(
        name="incident_get_channel_state",
        description="Read incident channel state model for a channel",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Discord channel ID"},
                "state": {
                    "type": "object",
                    "description": "Optional caller-supplied state snapshot",
                },
            },
            "required": ["channel_id"],
        },
    ),
    Tool(
        name="incident_set_channel_state",
        description="Set incident channel state model payload",
        inputSchema={
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Discord channel ID"},
                "state": {
                    "type": "object",
                    "description": "State model containing permission and slowdown controls",
                },
            },
            "required": ["channel_id", "state"],
        },
    ),
    Tool(
        name="incident_apply_lockdown",
        description="Apply incident lockdown to channels with confirm token enforcement",
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
        description="Rollback incident lockdown with reason and confirm token enforcement",
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
