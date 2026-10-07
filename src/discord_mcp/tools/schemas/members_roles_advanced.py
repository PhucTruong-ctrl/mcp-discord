from mcp.types import Tool


MEMBERS_ROLES_ADVANCED_TOOLS = [
    Tool(
        name="change_member_voice_state",
        description=(
            "Set a member's voice state in one call: move them to channel_id, "
            "server-mute and/or server-deafen them (dry-run by default). Omitting "
            "or emptying channel_id disconnects the member from voice — pass their "
            "current channel_id to change only the flags. Omitted mute/deafen are "
            "left unchanged. Discord decides permission (move_members, "
            "mute_members, deafen_members)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member ID"},
                "channel_id": {
                    "type": "string",
                    "description": (
                        "Voice channel ID; absent or empty disconnects the "
                        "member from voice"
                    ),
                },
                "mute": {
                    "type": "boolean",
                    "description": "Server mute flag; omitted leaves it unchanged",
                },
                "deafen": {
                    "type": "boolean",
                    "description": "Server deafen flag; omitted leaves it unchanged",
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return dry-run result",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "member_id"],
        },
    ),
    Tool(
        name="move_member_voice",
        description=(
            "Move a member to another voice channel, or disconnect them when "
            "channel_id is absent or empty (discord.py member.move_to(None); "
            "dry-run by default). Discord decides permission (move_members) and "
            "requires the member to already be connected to voice."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member ID"},
                "channel_id": {
                    "type": "string",
                    "description": (
                        "Voice channel ID; absent or empty disconnects the "
                        "member from voice"
                    ),
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return dry-run result",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "member_id"],
        },
    ),
    Tool(
        name="request_to_speak",
        description=(
            "Ask Discord to let a connected member speak on their stage channel "
            "(member.request_to_speak(); dry-run by default). The member must "
            "already be in voice — the tool rejects a member with no voice "
            "channel. Discord decides permission; a reason is required for the "
            "audit log."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member ID"},
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return dry-run result",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "member_id", "reason"],
        },
    ),
    Tool(
        name="get_member_voice_state",
        description=(
            "Read a member's voice state: connected channel, server mute/deafen, "
            "self mute/deafen, stream and video flags. Uses the gateway cache and "
            "falls back to a REST fetch. Every flag is null when the member is "
            "not in voice."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member ID"},
            },
            "required": ["server_id", "member_id"],
        },
    ),
    Tool(
        name="edit_member_profile",
        description=(
            "Edit a member's server-side profile: nickname, avatar, banner and/or "
            "bio (dry-run by default; at least one field required). "
            "avatar_url/banner_url are http(s) image URLs downloaded in-process, "
            "or null to remove the image; a null nickname or bio clears it. "
            "Discord decides permission: nickname needs manage_nicknames, while "
            "avatar/banner/bio can only be edited on the bot's own member."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "member_id": {"type": "string", "description": "Member ID"},
                "nickname": {
                    "type": ["string", "null"],
                    "description": "Nickname (max 32 chars); null or empty clears it",
                },
                "avatar_url": {
                    "type": ["string", "null"],
                    "description": "http(s) image URL; null removes the avatar",
                },
                "banner_url": {
                    "type": ["string", "null"],
                    "description": "http(s) image URL; null removes the banner",
                },
                "bio": {
                    "type": ["string", "null"],
                    "description": "Member bio; null clears it",
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return dry-run result",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "member_id"],
        },
    ),
    Tool(
        name="create_dm_channel",
        description=(
            "Open (or reuse) the DM channel with a user and return its ID. "
            "user_id may be any user, not necessarily a guild member. Read-only; "
            "the user's privacy settings decide whether Discord allows the DM."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User ID (any user, not necessarily a member)",
                },
            },
            "required": ["user_id"],
        },
    ),
    Tool(
        name="update_bot_profile",
        description=(
            "Update the bot's own profile — username, avatar and/or banner via "
            "ClientUser.edit (dry-run by default; at least one field required). "
            "avatar_url/banner_url are http(s) image URLs or null to remove the "
            "image. Discord decides username availability and image rules."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "username": {"type": "string", "description": "New bot username"},
                "avatar_url": {
                    "type": ["string", "null"],
                    "description": "http(s) image URL; null removes the avatar",
                },
                "banner_url": {
                    "type": ["string", "null"],
                    "description": "http(s) image URL; null removes the banner",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return dry-run result",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="set_role_icon",
        description=(
            "Set or clear a role's icon (dry-run by default). icon_url is an "
            "http(s) image URL; omitting it or passing null clears the icon "
            "(display_icon=None). Discord decides permission (manage_roles) and "
            "requires the ROLE_ICONS guild feature for role icons."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "role_id": {"type": "string", "description": "Role ID"},
                "icon_url": {
                    "type": ["string", "null"],
                    "description": (
                        "http(s) image URL; absent or null clears the role icon"
                    ),
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return dry-run result",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "role_id"],
        },
    ),
    Tool(
        name="reorder_roles",
        description=(
            "Reorder roles by mapping each role ID to its new integer position "
            "(dry-run by default). Positions must be contiguous — the Discord "
            "server rejects gaps between positions — and every key must be an "
            "existing role in the server. A reason is required for the audit log; "
            "Discord decides permission (manage_roles)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "positions": {
                    "type": "object",
                    "description": (
                        "Map of role ID to new integer position; positions must "
                        "be contiguous (the Discord server rejects gaps)"
                    ),
                    "additionalProperties": {"type": "integer"},
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return dry-run result",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Confirm token from dry-run",
                },
            },
            "required": ["server_id", "positions", "reason"],
        },
    ),
    Tool(
        name="get_role_details",
        description=(
            "Inspect role details: without role_id every role in the server "
            "sorted by position descending, otherwise just that role. Includes "
            "Role.tags, bot/integration/boost flags, display icon, decoded "
            "permission names and per-role member counts — member counts require "
            "manage_roles, so Discord rejects the whole call without it."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "role_id": {
                    "type": "string",
                    "description": "Role ID; omit to list every role",
                },
            },
            "required": ["server_id"],
        },
    ),
]
