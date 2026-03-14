from mcp.types import Tool


AUDIT_ANALYTICS_TOOLS = [
    Tool(
        name="get_audit_log",
        description="Fetch server audit log entries",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "limit": {"type": "number", "minimum": 1, "maximum": 1000},
                "action_type": {"type": "string"},
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="get_member_moderation_history",
        description="Get moderation history for a member",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "user_id": {"type": "string"},
                "limit": {"type": "number", "minimum": 1, "maximum": 1000},
            },
            "required": ["server_id", "user_id"],
        },
    ),
    Tool(
        name="get_channel_activity_summary",
        description="Summarize audit events for a channel",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "channel_id": {"type": "string"},
                "window_hours": {"type": "number", "minimum": 1},
            },
            "required": ["server_id", "channel_id"],
        },
    ),
    Tool(
        name="get_incident_timeline",
        description="Build incident timeline from audit events",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "channel_id": {"type": "string"},
                "user_id": {"type": "string"},
                "window_hours": {"type": "number", "minimum": 1},
            },
            "required": ["server_id", "window_hours"],
        },
    ),
    Tool(
        name="get_audit_actor_summary",
        description="Summarize audit events by actor",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "window_hours": {"type": "number", "minimum": 1},
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="check_audit_reason_compliance",
        description="Check whether moderation actions include reasons",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "window_hours": {"type": "number", "minimum": 1},
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="server_health_check",
        description="Run governance-focused server health checks",
        inputSchema={
            "type": "object",
            "properties": {"server_id": {"type": "string"}},
            "required": ["server_id"],
        },
    ),
    Tool(
        name="governance_evidence_packager",
        description="Package governance evidence for a review window",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string"},
                "window_hours": {"type": "number", "minimum": 1},
                "actor_id": {"type": "string"},
                "channel_id": {"type": "string"},
            },
            "required": ["server_id", "window_hours"],
        },
    ),
]
