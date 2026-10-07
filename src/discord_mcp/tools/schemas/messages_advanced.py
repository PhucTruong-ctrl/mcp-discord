from mcp.types import Tool


_COMPONENTS_SPEC = (
    'List of component specs: [{"type":"button","custom_id":"x","label":"L","emoji":"\U0001f525",'
    '"style":1,"disabled":false,"row":0}, {"type":"select","custom_id":"y","placeholder":"P",'
    '"options":[{"label":"a","value":"a","description":"d","emoji":"\U0001f525"}],'
    '"min_values":1,"max_values":1,"row":0}]. button.style is an int 1-5 '
    "(1 primary, 2 secondary, 3 success, 4 danger, 5 link); select needs custom_id and at "
    "least one option; unknown component types are rejected. Discord enforces at most 5 rows "
    "with 5 items each."
)

_DRY_RUN = {
    "type": "boolean",
    "description": "Return confirm token without acting when true",
    "default": True,
}
_CONFIRM_TOKEN = {
    "type": "string",
    "description": "Confirm token from the dry-run result; required when dry_run is false",
}


MESSAGES_ADVANCED_TOOLS = [
    Tool(
        name="send_message_with_files",
        description=(
            "Send a message with optional local-path and http(s)-URL attachments, stickers, "
            "mention controls and auto-delete. URL downloads run in-process (http/https only, "
            "no shell); Discord decides whether the bot may send to the channel."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Target channel ID"},
                "content": {"type": "string", "description": "Message text"},
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Local file paths to upload",
                },
                "file_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "http(s) URLs to download and upload; other schemes are rejected",
                },
                "sticker_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sticker IDs to attach (max 3)",
                },
                "tts": {"type": "boolean", "description": "Read aloud with text-to-speech"},
                "silent": {
                    "type": "boolean",
                    "description": "Send without push/desktop notification (highlight the channel)",
                },
                "suppress_embeds": {
                    "type": "boolean",
                    "description": "Suppress link embeds in this message",
                },
                "mention_everyone": {
                    "type": "boolean",
                    "description": "Allow @everyone/@here to ping",
                },
                "mention_author": {
                    "type": "boolean",
                    "description": "Allow this message to ping the referenced author",
                },
                "nonce": {"type": "string", "description": "Client nonce for the send"},
                "allowed_mention_roles": {
                    "type": ["array", "boolean"],
                    "items": {"type": "string"},
                    "description": "Role IDs that may ping (false = none, true = all)",
                },
                "allowed_mention_users": {
                    "type": ["array", "boolean"],
                    "items": {"type": "string"},
                    "description": "User IDs that may ping (false = none, true = all)",
                },
                "allowed_mention_echo": {
                    "type": "boolean",
                    "description": "Allow pinging the replied-to user",
                },
                "delete_after": {
                    "type": "number",
                    "description": "Seconds to wait before the bot deletes its own message",
                },
            },
            "required": ["server_id", "channel_id"],
        },
    ),
    Tool(
        name="send_components",
        description=(
            "Send a message carrying buttons and/or a select menu. " + _COMPONENTS_SPEC
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Target channel ID"},
                "components": {
                    "type": "array",
                    "description": _COMPONENTS_SPEC,
                },
            },
            "required": ["server_id", "channel_id", "components"],
        },
    ),
    Tool(
        name="send_poll",
        description=(
            "Send a Discord poll message. Answers: 2-10 entries (string or "
            '{"text": "...", "emoji": "..."}); duration_hours: 1-768 (Discord limit); '
            "custom poll emoji must be a <name:id> token or a {emoji, emojiId} object."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Target channel ID"},
                "question": {"type": "string", "description": "Poll question"},
                "answers": {
                    "type": "array",
                    "items": {"type": ["string", "object"]},
                    "description": "2-10 answers: text strings or {text, emoji} objects",
                    "minItems": 2,
                    "maxItems": 10,
                },
                "duration_hours": {
                    "type": "integer",
                    "description": "Poll duration in hours (1-768)",
                    "minimum": 1,
                    "maximum": 768,
                },
                "multiple": {
                    "type": "boolean",
                    "description": "Allow selecting more than one answer",
                    "default": False,
                },
                "layout_type": {
                    "type": "string",
                    "description": "Poll layout name (discord.py 2.7.1 exposes only 'default')",
                },
            },
            "required": ["server_id", "channel_id", "question", "answers", "duration_hours"],
        },
    ),
    Tool(
        name="get_poll_results",
        description=(
            "Fetch a message and read its poll: question, duration, total votes, per-answer "
            "rows and finalization. answers[].partial is true while the poll is still running "
            "(per-answer results are approximate until finalized)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Channel ID"},
                "message_id": {"type": "string", "description": "Message carrying the poll"},
            },
            "required": ["server_id", "channel_id", "message_id"],
        },
    ),
    Tool(
        name="forward_message",
        description=(
            "Forward a message to another channel (Discord's forward, not a copy-paste). "
            "Discord decides whether the bot may post in the destination."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Channel holding the message"},
                "message_id": {"type": "string", "description": "Message to forward"},
                "destination_channel_id": {
                    "type": "string",
                    "description": "Channel to forward into",
                },
            },
            "required": ["server_id", "channel_id", "message_id", "destination_channel_id"],
        },
    ),
    Tool(
        name="pin_message",
        description=(
            "Pin a message in its channel. Requires manage_messages (Discord decides). "
            "dry_run by default; confirm_token required to pin."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Channel ID"},
                "message_id": {"type": "string", "description": "Message to pin"},
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": _DRY_RUN,
                "confirm_token": _CONFIRM_TOKEN,
            },
            "required": ["server_id", "channel_id", "message_id"],
        },
    ),
    Tool(
        name="unpin_message",
        description=(
            "Remove a pin from a message. Requires manage_messages (Discord decides). "
            "dry_run by default; confirm_token required to unpin."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Channel ID"},
                "message_id": {"type": "string", "description": "Message to unpin"},
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": _DRY_RUN,
                "confirm_token": _CONFIRM_TOKEN,
            },
            "required": ["server_id", "channel_id", "message_id"],
        },
    ),
    Tool(
        name="clear_message_reactions",
        description=(
            "Remove every reaction from a message (single-emoji removal already exists "
            "elsewhere). Requires manage_messages (Discord decides). Discord's "
            "clear-all-reactions endpoint takes no audit reason, so reason is echoed in the "
            "payload only. dry_run by default; confirm_token required to clear."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Channel ID"},
                "message_id": {"type": "string", "description": "Message to clear"},
                "reason": {
                    "type": "string",
                    "description": "Reason echoed in the payload (Discord records no audit entry)",
                },
                "dry_run": _DRY_RUN,
                "confirm_token": _CONFIRM_TOKEN,
            },
            "required": ["server_id", "channel_id", "message_id"],
        },
    ),
    Tool(
        name="get_reaction_users",
        description=(
            "List users who reacted with a given emoji on a message. Errors when the message "
            "has no reactions or the emoji is absent. Emoji: unicode, <name:id> token, or a "
            "bare custom-emoji name resolved against this server."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Channel ID"},
                "message_id": {"type": "string", "description": "Message ID"},
                "emoji": {
                    "type": "string",
                    "description": "Reaction emoji (unicode, <name:id>, or guild emoji name)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max users to return (omit for every reactor)",
                    "minimum": 1,
                },
            },
            "required": ["server_id", "channel_id", "message_id", "emoji"],
        },
    ),
    Tool(
        name="create_thread_from_message",
        description=(
            "Create a public thread attached to an existing message. Requires "
            "create_public_threads (Discord decides). dry_run by default; confirm_token "
            "required to create."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Channel ID"},
                "message_id": {"type": "string", "description": "Message to start the thread from"},
                "name": {"type": "string", "description": "Thread name"},
                "auto_archive_duration": {
                    "type": "integer",
                    "description": "Auto-archive minutes: one of 60, 1440, 4320, 10080",
                    "enum": [60, 1440, 4320, 10080],
                },
                "slowmode_delay": {
                    "type": "integer",
                    "description": "Slowmode seconds (0-21600)",
                    "minimum": 0,
                    "maximum": 21600,
                },
                "reason": {"type": "string", "description": "Audit log reason"},
                "dry_run": _DRY_RUN,
                "confirm_token": _CONFIRM_TOKEN,
            },
            "required": ["server_id", "channel_id", "message_id", "name"],
        },
    ),
    Tool(
        name="send_typing",
        description=(
            "Show the typing indicator in a channel for the duration of the call (Discord "
            "shows it up to ~10 seconds). Cosmetic only; text channels and threads."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {"type": "string", "description": "Target channel ID"},
            },
            "required": ["server_id", "channel_id"],
        },
    ),
]
