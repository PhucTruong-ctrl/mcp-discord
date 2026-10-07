from mcp.types import Tool


PERMISSION_INTEL_TOOLS = [
    Tool(
        name="get_role_permissions",
        description=(
            "Return role permission bitfields (int) with decoded permission names. "
            "Omit role_id to list every role including the @everyone role; pass role_id "
            "for a single role. Use it to answer 'which role grants MENTION_EVERYONE'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "role_id": {
                    "type": "string",
                    "description": "Optional role ID; omit to list all roles",
                },
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="compute_member_permissions",
        description=(
            "Resolve a member's effective permission bitfield and report which layer "
            "decided each permission (administrator, base_role, everyone_overwrite, "
            "role_overwrite:<roleId>, member_overwrite, suffixed :allow/:deny). Pass "
            "channel_id to include that channel's overwrites. Overwrite resolution only: "
            "discord.py's per-channel-type flag stripping is not applied, and category "
            "overwrites are reported in categoryOverwrites rather than inherited."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member/user ID"},
                "channel_id": {
                    "type": "string",
                    "description": "Optional channel ID for channel-level resolution",
                },
            },
            "required": ["server_id", "member_id"],
        },
    ),
]
