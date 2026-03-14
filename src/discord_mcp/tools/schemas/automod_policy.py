from mcp.types import Tool


AUTOMOD_POLICY_TOOLS = [
    Tool(
        name="automod_validate_ruleset",
        description="Validate caller-supplied AutoMod ruleset model",
        inputSchema={
            "type": "object",
            "properties": {
                "ruleset": {
                    "type": "object",
                    "description": "Caller-supplied ruleset model",
                    "properties": {
                        "name": {"type": "string"},
                        "rules": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "trigger_type": {"type": "string"},
                                    "trigger_metadata": {"type": "object"},
                                    "actions": {
                                        "type": "array",
                                        "items": {"type": "object"},
                                    },
                                    "enabled": {"type": "boolean"},
                                },
                                "required": ["name", "trigger_type", "actions"],
                            },
                        },
                    },
                    "required": ["name", "rules"],
                }
            },
            "required": ["ruleset"],
        },
    ),
    Tool(
        name="automod_get_ruleset",
        description="Return caller-supplied ruleset model for a guild",
        inputSchema={
            "type": "object",
            "properties": {
                "guild_id": {"type": "string", "description": "Discord guild ID"},
                "ruleset": {
                    "type": "object",
                    "description": "Caller-supplied ruleset model",
                    "properties": {
                        "name": {"type": "string"},
                        "rules": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["name", "rules"],
                },
            },
            "required": ["guild_id", "ruleset"],
        },
    ),
    Tool(
        name="automod_apply_ruleset",
        description="Apply caller-supplied ruleset with reason and confirm token enforcement",
        inputSchema={
            "type": "object",
            "properties": {
                "guild_id": {"type": "string", "description": "Discord guild ID"},
                "ruleset": {
                    "type": "object",
                    "description": "Caller-supplied ruleset model",
                    "properties": {
                        "name": {"type": "string"},
                        "rules": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["name", "rules"],
                },
                "reason": {"type": "string", "description": "Required audit reason"},
                "dry_run": {
                    "type": "boolean",
                    "description": "Return confirm token when true",
                    "default": True,
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Required for execute path when dry_run is false",
                },
            },
            "required": ["guild_id", "ruleset", "reason"],
        },
    ),
    Tool(
        name="automod_rollback_ruleset",
        description="Rollback ruleset state with reason and confirm token enforcement",
        inputSchema={
            "type": "object",
            "properties": {
                "guild_id": {"type": "string", "description": "Discord guild ID"},
                "ruleset_name": {
                    "type": "string",
                    "description": "Ruleset identifier to rollback",
                },
                "reason": {"type": "string", "description": "Required audit reason"},
                "dry_run": {
                    "type": "boolean",
                    "description": "Return confirm token when true",
                    "default": True,
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Required for execute path when dry_run is false",
                },
            },
            "required": ["guild_id", "ruleset_name", "reason"],
        },
    ),
]
