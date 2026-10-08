from mcp.types import Tool


INVITES_MEMBERSHIP_TOOLS = [
    Tool(
        name="create_invite",
        description=(
            "Create an instant invite for a channel. Discord enforces the "
            "CREATE_INSTANT_INVITE permission and invite limits; target_type=stream "
            "also needs target_user and embedded_application needs "
            "target_application_id. dry_run by default; pass the confirm_token "
            "from the dry run to execute."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {
                    "type": "string",
                    "description": "Channel to create the invite on",
                },
                "max_age": {
                    "type": "integer",
                    "description": "Lifetime in seconds (0 = never expires)",
                    "minimum": 0,
                },
                "max_uses": {
                    "type": "integer",
                    "description": "Maximum uses (0 = unlimited)",
                    "minimum": 0,
                },
                "temporary": {
                    "type": "boolean",
                    "description": "Kick on disconnect (default false)",
                },
                "unique": {
                    "type": "boolean",
                    "description": "Force a unique code (default true)",
                },
                "target_type": {
                    "type": "string",
                    "enum": ["stream", "embedded_application"],
                    "description": "Voice-channel invite target",
                },
                "target_user": {
                    "type": "string",
                    "description": "Featured streamer user ID (stream)",
                },
                "target_application_id": {
                    "type": "string",
                    "description": "App ID (embedded_application target)",
                },
                "guest": {
                    "type": "boolean",
                    "description": "Guest invite (default false)",
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview only; returns confirm token (default true)",
                    "default": True,
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from the dry run",
                },
            },
            "required": ["server_id", "channel_id"],
        },
    ),
    Tool(
        name="list_invites",
        description=(
            "List instant invites for a server (needs MANAGE_GUILD) or for one "
            "channel when channel_id is given (needs MANAGE_CHANNELS). Discord "
            "enforces the permission; Forbidden surfaces when the bot lacks it."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {
                    "type": "string",
                    "description": "Only list invites for this channel",
                },
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="delete_invite",
        description=(
            "Revoke an instant invite by code or discord.gg URL. Discord enforces "
            "MANAGE_CHANNELS and the revocation is irreversible: the code stops "
            "working immediately. reason is required; dry_run by default, pass "
            "the confirm_token from the dry run to execute."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "invite_code": {
                    "type": "string",
                    "description": "Invite code or discord.gg URL",
                },
                "reason": {
                    "type": "string",
                    "description": "Audit log reason (required)",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview only; returns confirm token (default true)",
                    "default": True,
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from the dry run",
                },
            },
            "required": ["server_id", "invite_code", "reason"],
        },
    ),
    Tool(
        name="list_bans",
        description=(
            "List a server's bans with their reasons. Discord enforces the "
            "BAN_MEMBERS permission; Forbidden surfaces when the bot lacks it."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum bans to return (default and cap 1000)",
                },
                "before": {
                    "type": "string",
                    "description": "Only return bans created before this snowflake id",
                },
                "after": {
                    "type": "string",
                    "description": "Only return bans created after this snowflake id",
                },
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="get_ban",
        description=(
            "Fetch the ban record and reason for one user (Discord enforces "
            "BAN_MEMBERS). A user who is not banned surfaces as a ValueError."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "user_id": {"type": "string", "description": "User ID to look up"},
            },
            "required": ["server_id", "user_id"],
        },
    ),
    Tool(
        name="search_members",
        description=(
            "Search members by name/nickname prefix, or fetch specific members "
            "by user_ids (the two are mutually exclusive). Requires the members "
            "intent, which is enabled in server.py; Discord answers from the "
            "gateway, so results reflect live membership."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "query": {
                    "type": "string",
                    "description": "Name or nickname prefix to match",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum members to return (default 5, cap 100)",
                },
                "user_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fetch these member IDs instead of searching",
                },
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="get_role_member_counts",
        description=(
            "Member count per role, sorted by role position (Discord enforces "
            "MANAGE_ROLES). Roles missing from the cache report null name and "
            "position."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="estimate_pruned_members",
        description=(
            "Estimate how many members a prune would remove after `days` days "
            "of inactivity; read-only, nothing is pruned. Discord requires "
            "MANAGE_GUILD and KICK_MEMBERS, and returns no count when prune "
            "counting is disabled (prunable=false, count=null)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "days": {
                    "type": "integer",
                    "description": "Days of inactivity (1-30)",
                    "minimum": 1,
                    "maximum": 30,
                },
                "role_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only members holding these roles count",
                },
            },
            "required": ["server_id", "days"],
        },
    ),
]
