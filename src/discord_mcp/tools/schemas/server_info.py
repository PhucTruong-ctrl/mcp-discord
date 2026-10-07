from mcp.types import Tool


SERVER_INFO_TOOLS = [
    Tool(
        name="get_server_info",
        description="Get information about a Discord server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server (guild) ID",
                }
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="get_channels",
        description="Get a list of all channels in a Discord server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server (guild) ID",
                }
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="list_members",
        description="Get a list of members in a server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server (guild) ID",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of members to fetch",
                    "minimum": 1,
                    "maximum": 1000,
                },
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="list_servers",
        description="Get a list of all Discord servers the bot has access to with their details such as name, id, member count, and creation date.",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="update_guild",
        description=(
            "Update guild settings (discord.py Guild.edit): name, description, "
            "preferred_locale, vanity_code, verification_level, explicit_content_filter, "
            "default_notifications, mfa_level, community, discoverable, invites_disabled, "
            "widget_enabled, premium_progress_bar_enabled, raid_alerts_disabled, "
            "afk_channel/afk_timeout, system_channel(+flags), rules_channel, "
            "public_updates_channel, safety_alerts_channel, widget_channel, owner, and the "
            "image fields icon/banner/splash/discovery_splash (http(s) URL, data URI or local "
            "path). Not settable through the API: server-profile traits, games, banner colour, "
            "private profile and the server tag. Unknown fields are rejected."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Discord server (guild) ID",
                },
                "name": {"type": "string", "description": "Server name"},
                "description": {
                    "type": ["string", "null"],
                    "description": "Server description; null clears it",
                },
                "preferred_locale": {
                    "type": "string",
                    "description": "Community guild locale, e.g. en-US, vi",
                },
                "vanity_code": {"type": "string", "description": "Vanity invite code"},
                "verification_level": {
                    "type": ["string", "number"],
                    "description": "none/low/medium/high/highest or 0-4",
                },
                "explicit_content_filter": {
                    "type": ["string", "number"],
                    "description": "disabled/no_role/all_members or 0-2",
                },
                "default_notifications": {
                    "type": ["string", "number"],
                    "description": "all_messages/only_mentions or 0-1",
                },
                "mfa_level": {
                    "type": ["string", "number"],
                    "description": "none/elevated or 0-1 (requires the owner's 2FA)",
                },
                "community": {
                    "type": "boolean",
                    "description": "Toggle the COMMUNITY feature",
                },
                "discoverable": {
                    "type": "boolean",
                    "description": "Toggle server discovery",
                },
                "invites_disabled": {
                    "type": "boolean",
                    "description": "Pause new invites",
                },
                "widget_enabled": {
                    "type": "boolean",
                    "description": "Toggle the server widget",
                },
                "premium_progress_bar_enabled": {"type": "boolean"},
                "raid_alerts_disabled": {"type": "boolean"},
                "afk_channel": {
                    "type": ["string", "null"],
                    "description": "AFK voice channel id or name; null clears it",
                },
                "afk_timeout": {
                    "type": "number",
                    "description": "60, 300, 900, 1800 or 3600 seconds",
                },
                "system_channel": {"type": ["string", "null"]},
                "system_channel_flags": {
                    "type": ["number", "array"],
                    "description": "Bitfield int or array of flag names",
                },
                "rules_channel": {"type": ["string", "null"]},
                "public_updates_channel": {"type": ["string", "null"]},
                "safety_alerts_channel": {"type": ["string", "null"]},
                "widget_channel": {"type": ["string", "null"]},
                "owner": {
                    "type": "string",
                    "description": "Transfer ownership (user id)",
                },
                "icon": {
                    "type": ["string", "null"],
                    "description": "http(s) URL, data URI or local path; null removes it",
                },
                "banner": {
                    "type": ["string", "null"],
                    "description": "Needs the BANNER guild feature; null removes it",
                },
                "splash": {
                    "type": ["string", "null"],
                    "description": "Needs INVITE_SPLASH",
                },
                "discovery_splash": {
                    "type": ["string", "null"],
                    "description": "Needs DISCOVERABLE",
                },
                "invites_disabled_until": {"type": ["string", "null"]},
                "dms_disabled_until": {"type": ["string", "null"]},
                "reason": {
                    "type": "string",
                    "description": "Audit log reason",
                },
            },
            "required": ["server_id"],
        },
    ),
]
