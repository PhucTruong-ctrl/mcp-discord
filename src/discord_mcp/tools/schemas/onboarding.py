from mcp.types import Tool


ONBOARDING_TOOLS = [
    Tool(
        name="get_guild_welcome_screen",
        description="Get the guild welcome screen configuration",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"}
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="update_guild_welcome_screen",
        description="Update the guild welcome screen payload",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "welcome_screen": {
                    "type": "object",
                    "description": "Welcome screen payload",
                },
                "reason": {"type": "string", "description": "Audit log reason"},
            },
            "required": ["server_id", "welcome_screen"],
        },
    ),
    Tool(
        name="get_guild_onboarding",
        description="Get guild onboarding configuration",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"}
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="update_guild_onboarding",
        description="Update guild onboarding configuration",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "onboarding": {
                    "type": "object",
                    "description": "Onboarding payload",
                },
                "reason": {"type": "string", "description": "Audit log reason"},
            },
            "required": ["server_id", "onboarding"],
        },
    ),
    Tool(
        name="dynamic_role_provision",
        description="Apply role add/remove operations from caller-provided ruleset",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "user_id": {"type": "string"},
                "facts": {"type": "object"},
                "ruleset": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "condition": {"type": "string"},
                            "role_id": {"type": "string"},
                            "op": {"type": "string", "enum": ["add", "remove"]},
                        },
                        "required": ["condition", "role_id", "op"],
                    },
                },
                "reason": {"type": "string"},
            },
            "required": ["server_id", "user_id", "ruleset"],
        },
    ),
    Tool(
        name="verification_gate_orchestrator",
        description="Evaluate caller-provided verification gates",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "user_id": {"type": "string"},
                "gates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "membership_age",
                                    "has_role",
                                    "manual_approve",
                                ],
                            },
                            "config": {"type": "object"},
                        },
                        "required": ["type", "config"],
                    },
                },
                "mode": {"type": "string", "enum": ["all", "any"]},
                "facts": {"type": "object"},
            },
            "required": ["gates", "mode"],
        },
    ),
    Tool(
        name="progressive_access_unlock",
        description="Compute unlocks from caller-provided policy and facts",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "user_id": {"type": "string"},
                "policy": {
                    "type": "object",
                    "properties": {
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "unlocks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["role", "channel"],
                                    },
                                    "id": {"type": "string"},
                                    "requires": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["type", "id"],
                            },
                        },
                    },
                    "required": ["requirements", "unlocks"],
                },
                "facts": {"type": "object"},
            },
            "required": ["policy"],
        },
    ),
    Tool(
        name="onboarding_friction_audit",
        description="Compute onboarding friction metrics from caller-provided stage stats",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "window_days": {"type": "number", "minimum": 1},
                "stage_stats": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "stage": {"type": "string"},
                            "entered": {"type": "number"},
                            "completed": {"type": "number"},
                        },
                        "required": ["stage", "entered", "completed"],
                    },
                },
            },
            "required": ["server_id", "window_days"],
        },
    ),
]
