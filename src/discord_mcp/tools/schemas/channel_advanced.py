from mcp.types import Tool


CHANNEL_ADVANCED_TOOLS = [
    Tool(
        name="clone_channel",
        description=(
            "Clone a channel (text, voice, category, forum or announcement) with the "
            "same properties. Requires Manage Channels; Discord decides the new "
            "channel's position and id. Optional name defaults to the source name."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {"type": "string", "description": "Channel to clone"},
                "name": {
                    "type": "string",
                    "description": "Name for the clone (defaults to the source channel name)",
                },
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for the clone",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without cloning; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "channel_id"],
        },
    ),
    Tool(
        name="create_announcement_channel",
        description=(
            "Create an announcement (news) channel. Requires Manage Channels; "
            "Discord decides the final id and position. Optional category_id places "
            "the channel under a category (permission overwrites sync from the "
            "category)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "name": {"type": "string", "description": "Channel name"},
                "category_id": {
                    "type": "string",
                    "description": "Optional category ID to place the channel in",
                },
                "topic": {"type": "string", "description": "Optional channel topic"},
                "nsfw": {"type": "boolean", "description": "Optional NSFW flag"},
                "slowmode_delay": {
                    "type": "integer",
                    "description": "Optional slowmode in seconds (max 21600)",
                },
                "default_auto_archive_duration": {
                    "type": "integer",
                    "description": "Default thread auto-archive minutes: 60, 1440, 4320 or 10080",
                },
                "default_thread_slowmode_delay": {
                    "type": "integer",
                    "description": "Optional default slowmode for new threads in seconds",
                },
                "position": {
                    "type": "integer",
                    "description": "Optional channel position",
                },
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for the creation",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without creating; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="create_stage_channel",
        description=(
            "Create a stage channel. Requires Manage Channels; Discord decides the "
            "final id and position. video_quality_mode is 1 (auto) or 2 (full); "
            "rtc_region is a Discord voice region string (e.g. us-central) or null "
            "for automatic."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "name": {"type": "string", "description": "Channel name"},
                "category_id": {
                    "type": "string",
                    "description": "Optional category ID to place the channel in",
                },
                "bitrate": {
                    "type": "integer",
                    "description": "Optional voice bitrate in bits per second",
                },
                "user_limit": {
                    "type": "integer",
                    "description": "Optional user limit (0 = unlimited)",
                },
                "rtc_region": {
                    "type": "string",
                    "description": "Optional voice region (e.g. us-central); omitted = automatic",
                },
                "video_quality_mode": {
                    "type": "integer",
                    "description": "Video quality: 1 (auto) or 2 (full)",
                },
                "nsfw": {"type": "boolean", "description": "Optional NSFW flag"},
                "position": {
                    "type": "integer",
                    "description": "Optional channel position",
                },
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for the creation",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without creating; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="follow_channel",
        description=(
            "Follow an announcement channel into a text channel via webhook. The "
            "source (channel_id) must be an announcement channel; the target "
            "(webhook_channel_id) must be a text channel in the same server. "
            "Requires Manage Webhooks on the target — Discord decides the "
            "permission."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {
                    "type": "string",
                    "description": "Announcement channel to follow",
                },
                "webhook_channel_id": {
                    "type": "string",
                    "description": "Text channel that will receive the followed posts",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without following; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "channel_id", "webhook_channel_id"],
        },
    ),
    Tool(
        name="sync_channel_permissions",
        description=(
            "Mirror a category's permission overwrites onto every child channel — "
            "a client-side helper matching what Discord's UI does on category "
            "permission sync, NOT a single Discord API call: each differing "
            "overwrite is written with its own set_permissions request (and child "
            "overwrites missing from the category are deleted). Requires Manage "
            "Roles/Manage Channels on each child; Discord decides whether the bot "
            "may write. The dry run reports a per-channel diff of previous vs "
            "applied overwrites."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "category_id": {
                    "type": "string",
                    "description": "Category whose overwrites will be mirrored",
                },
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for each overwrite change",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview the per-channel diff without writing; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "category_id"],
        },
    ),
    Tool(
        name="set_voice_channel_status",
        description=(
            "Set the status string shown on a voice or stage channel (e.g. "
            "'Recording'). Only voice and stage channels support status; Discord "
            "truncates the value at 500 characters and requires Connect permission "
            "on the channel."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {
                    "type": "string",
                    "description": "Voice or stage channel ID",
                },
                "status": {
                    "type": "string",
                    "description": "Status text, 1-500 characters (Discord truncates at 500)",
                },
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for the change",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without changing the status; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "channel_id", "status"],
        },
    ),
]
