from mcp.types import Tool


EMOJI_STICKER_SOUNDBOARD_TOOLS = [
    Tool(
        name="create_emoji",
        description=(
            "Create a custom emoji on the server (dry-run by default). image_url must "
            "be an http/https URL. Discord decides permission (Manage Emojis & "
            "Stickers) and enforces the guild emoji cap."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "name": {"type": "string", "description": "Emoji name (2-32 chars)"},
                "image_url": {
                    "type": "string",
                    "description": "http/https URL of the PNG/GIF image",
                },
                "role_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Roles allowed to use the emoji; omitted or empty means "
                        "everyone"
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
            "required": ["server_id", "name", "image_url"],
        },
    ),
    Tool(
        name="edit_emoji",
        description=(
            "Rename a custom emoji and/or restrict which roles may use it (dry-run "
            "by default). Discord decides permission (Manage Emojis & Stickers)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "emoji_id": {"type": "string", "description": "Emoji ID"},
                "name": {"type": "string", "description": "New emoji name"},
                "role_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Roles allowed to use the emoji; an empty list makes it "
                        "available to everyone"
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
            "required": ["server_id", "emoji_id"],
        },
    ),
    Tool(
        name="delete_emoji",
        description=(
            "Delete a custom emoji permanently (dry-run by default; irreversible). "
            "reason is required for the audit log. Discord decides permission "
            "(Manage Emojis & Stickers)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "emoji_id": {"type": "string", "description": "Emoji ID"},
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
            "required": ["server_id", "emoji_id", "reason"],
        },
    ),
    Tool(
        name="create_application_emoji",
        description=(
            "Create an emoji on the application itself - usable in every guild the "
            "bot is in, not owned by any one server (dry-run by default). Discord "
            "caps applications at 50 emoji and decides permission."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Emoji name (2-32 chars)"},
                "image_url": {
                    "type": "string",
                    "description": "http/https URL of the PNG/GIF image",
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
            "required": ["name", "image_url"],
        },
    ),
    Tool(
        name="edit_application_emoji",
        description=(
            "Rename an application emoji (dry-run by default). The emoji belongs to "
            "the application, not to a guild; Discord decides permission."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "emoji_id": {"type": "string", "description": "Application emoji ID"},
                "name": {"type": "string", "description": "New emoji name"},
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
            "required": ["emoji_id"],
        },
    ),
    Tool(
        name="delete_application_emoji",
        description=(
            "Delete an application emoji permanently (dry-run by default; "
            "irreversible). There is no reason parameter: Discord's "
            "application-emoji API accepts no audit-log reason."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "emoji_id": {"type": "string", "description": "Application emoji ID"},
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
            "required": ["emoji_id"],
        },
    ),
    Tool(
        name="list_application_emojis",
        description=(
            "List the application's own emoji (account-level, not guild-level). "
            "No confirmation gate."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="create_sticker",
        description=(
            "Create a guild sticker (dry-run by default). file_path is a local file "
            "readable by the MCP server (PNG, 32-512 KB); emoji must be a unicode "
            "emoji; reason is required for the audit log. Discord decides permission "
            "(Manage Expressions)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "name": {"type": "string", "description": "Sticker name (2-30 chars)"},
                "description": {
                    "type": "string",
                    "description": "Sticker description (up to 200 chars)",
                },
                "emoji": {
                    "type": "string",
                    "description": "Unicode emoji associated with the sticker",
                },
                "file_path": {
                    "type": "string",
                    "description": "Local path of the sticker image file",
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
            "required": ["server_id", "name", "description", "emoji", "file_path", "reason"],
        },
    ),
    Tool(
        name="edit_sticker",
        description=(
            "Edit a guild sticker's name, description and/or emoji (dry-run by "
            "default). Only the supplied fields are changed; Discord decides "
            "permission (Manage Expressions)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "sticker_id": {"type": "string", "description": "Sticker ID"},
                "name": {"type": "string", "description": "New sticker name"},
                "description": {
                    "type": "string",
                    "description": "New sticker description",
                },
                "emoji": {
                    "type": "string",
                    "description": "New unicode emoji for the sticker",
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
            "required": ["server_id", "sticker_id"],
        },
    ),
    Tool(
        name="delete_sticker",
        description=(
            "Delete a guild sticker permanently (dry-run by default; irreversible). "
            "reason is required for the audit log. Discord decides permission "
            "(Manage Expressions)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "sticker_id": {"type": "string", "description": "Sticker ID"},
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
            "required": ["server_id", "sticker_id", "reason"],
        },
    ),
    Tool(
        name="list_stickers",
        description=(
            "Fetch the guild's stickers straight from the API (fresh read, not the "
            "cache). No confirmation gate."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="create_soundboard_sound",
        description=(
            "Create a soundboard sound (dry-run by default). sound_url must be an "
            "http/https URL of an MP3/OGG up to 5.2 seconds - Discord enforces "
            "format and length. volume must be between 0.0 and 1.0. Discord decides "
            "permission (Create Expressions)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "name": {"type": "string", "description": "Sound name (2-32 chars)"},
                "sound_url": {
                    "type": "string",
                    "description": "http/https URL of the MP3/OGG sound",
                },
                "volume": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Playback volume between 0.0 and 1.0 (default 1.0)",
                },
                "emoji": {
                    "type": "string",
                    "description": (
                        "Emoji shown with the sound: unicode, bare custom-emoji name "
                        "or <:name:id> token"
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
            "required": ["server_id", "name", "sound_url"],
        },
    ),
    Tool(
        name="list_soundboard_sounds",
        description=(
            "Fetch the guild's soundboard sounds straight from the API (fresh read, "
            "not the cache). No confirmation gate."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="edit_soundboard_sound",
        description=(
            "Edit a soundboard sound's name, volume and/or emoji (dry-run by "
            "default). Discord decides permission (Manage Expressions)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "sound_id": {"type": "string", "description": "Soundboard sound ID"},
                "name": {"type": "string", "description": "New sound name"},
                "volume": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Playback volume between 0.0 and 1.0",
                },
                "emoji": {
                    "type": "string",
                    "description": (
                        "Emoji shown with the sound: unicode, bare custom-emoji name "
                        "or <:name:id> token"
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
            "required": ["server_id", "sound_id"],
        },
    ),
    Tool(
        name="delete_soundboard_sound",
        description=(
            "Delete a soundboard sound permanently (dry-run by default; "
            "irreversible). reason is required for the audit log. Discord decides "
            "permission (Manage Expressions)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "sound_id": {"type": "string", "description": "Soundboard sound ID"},
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
            "required": ["server_id", "sound_id", "reason"],
        },
    ),
    Tool(
        name="send_soundboard_sound",
        description=(
            "Play a soundboard sound in a voice channel. The channel must be a voice "
            "channel and the sound must belong to this server; Discord decides "
            "permission (speak + use_soundboard). No gate - the sound plays "
            "immediately."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Voice channel ID"},
                "sound_id": {"type": "string", "description": "Soundboard sound ID"},
            },
            "required": ["server_id", "channel_id", "sound_id"],
        },
    ),
]
