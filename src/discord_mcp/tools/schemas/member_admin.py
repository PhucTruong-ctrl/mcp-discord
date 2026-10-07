from mcp.types import Tool


MEMBER_ADMIN_TOOLS = [
    Tool(
        name="set_member_roles",
        description=(
            "Replace a member's entire role set in one call (discord.py Member.edit(roles=...)). "
            "role_ids=[] removes every assignable role. @everyone is implicit and ignored; roles "
            "managed by an integration are rejected. dry_run by default; pass confirm_token to "
            "apply. Reports added/removed/ignored and whether Discord's final state matches."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member/user ID"},
                "role_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Complete role set to apply (empty array clears roles)",
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {
                    "type": "boolean",
                    "description": "Return the plan + confirm token without applying (default true)",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Required when dry_run is false",
                },
            },
            "required": ["server_id", "member_id", "role_ids"],
        },
    ),
    Tool(
        name="set_member_nickname",
        description=(
            "Set or clear one member's server nickname (discord.py Member.edit(nick=...)). "
            "Pass nickname='' or null to remove the nickname. Needs MANAGE_NICKNAMES for "
            "other members, CHANGE_NICKNAME for the bot's own record; nickname max 32 chars."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member/user ID"},
                "nickname": {
                    "type": ["string", "null"],
                    "description": "New nickname; empty string or null removes it",
                },
                "reason": {"type": "string", "description": "Audit log reason"},
            },
            "required": ["server_id", "member_id", "nickname"],
        },
    ),
]
