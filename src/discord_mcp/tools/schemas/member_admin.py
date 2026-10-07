from mcp.types import Tool


MEMBER_ADMIN_TOOLS = [
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
