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
                                    "exempt_roles": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Role ids or names exempt from the rule (max 20)",
                                    },
                                    "exempt_channels": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Channel ids or names exempt from the rule (max 50)",
                                    },
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
        description="Fetch AutoMod ruleset(s) for a guild from Discord",
        inputSchema={
            "type": "object",
            "properties": {
                "guild_id": {"type": "string", "description": "Discord guild ID"},
                "ruleset_name": {
                    "type": "string",
                    "description": "Optional name to filter rules by",
                },
            },
            "required": ["guild_id"],
        },
    ),
    Tool(
        name="automod_apply_ruleset",
        description=(
            "Apply caller-supplied ruleset with reason and confirm token enforcement. Each "
            "rule supports trigger_type keyword (keyword_filter / regex_patterns / allow_list), "
            "keyword_preset, mention_spam and member_profile, actions block_message / "
            "send_alert_message / timeout / block_member_interaction, plus exempt_roles and "
            "exempt_channels given as ids or names."
        ),
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
        description="Rollback is not supported without persistent state tracking. "
        "Returns an explicit unsupported response. "
        "Use automod_apply_ruleset with a prior ruleset snapshot for manual rollback.",
        inputSchema={
            "type": "object",
            "properties": {
                "guild_id": {
                    "type": "string",
                    "description": "Discord guild ID for capability-check/execute requests",
                },
                "ruleset_name": {
                    "type": "string",
                    "description": "Optional ruleset label for operator context",
                },
                "reason": {
                    "type": "string",
                    "description": "Required audit reason",
                },
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
            "required": ["guild_id", "reason"],
        },
    ),
]
