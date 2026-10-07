from mcp.types import Tool


SCHEDULED_STAGE_TOOLS = [
    Tool(
        name="create_scheduled_event",
        description=(
            "Create a guild scheduled event (dry-run by default). entity_type is "
            "stage_instance (channel_id must be a stage channel), voice_channel "
            "(voice channel) or external (no channel_id; location and end_time "
            "required). start_time/end_time are ISO-8601 with a UTC offset (Z or "
            "+00:00). Discord decides permission (Manage Events) and enforces the "
            "entity/channel combinations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "name": {"type": "string", "description": "Event name"},
                "entity_type": {
                    "type": "string",
                    "enum": ["stage_instance", "voice_channel", "external"],
                    "description": (
                        "What the event is attached to: a stage channel, a voice "
                        "channel, or an external location"
                    ),
                },
                "start_time": {
                    "type": "string",
                    "description": (
                        "ISO-8601 start time with offset, e.g. "
                        "2026-12-01T18:00:00Z or 2026-12-01T18:00:00+00:00"
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Optional event description",
                },
                "channel_id": {
                    "type": "string",
                    "description": (
                        "Stage/voice channel the event runs in; required for "
                        "stage_instance and voice_channel, forbidden for external"
                    ),
                },
                "end_time": {
                    "type": "string",
                    "description": (
                        "ISO-8601 end time with offset; required for external events"
                    ),
                },
                "privacy_level": {
                    "type": "string",
                    "enum": ["guild_only"],
                    "description": (
                        "Event privacy level (discord.py 2.7.1 defines guild_only)"
                    ),
                },
                "image_url": {
                    "type": "string",
                    "description": "Optional http(s) URL of the event cover image",
                },
                "location": {
                    "type": "string",
                    "description": (
                        "Physical location; required for external events, forbidden "
                        "otherwise"
                    ),
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
            "required": ["server_id", "name", "entity_type", "start_time"],
        },
    ),
    Tool(
        name="get_scheduled_event",
        description=(
            "Fetch one guild scheduled event by id. No confirmation gate. "
            "with_user_count asks Discord for the subscriber count; userCount is "
            "omitted from the row unless requested."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "event_id": {"type": "string", "description": "Scheduled event ID"},
                "with_user_count": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Also request Discord's subscriber count for the event"
                    ),
                },
            },
            "required": ["server_id", "event_id"],
        },
    ),
    Tool(
        name="list_scheduled_events",
        description=(
            "List the server's guild scheduled events, soonest first. No "
            "confirmation gate. with_user_count asks Discord for subscriber "
            "counts; userCount is omitted from each row unless requested."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "with_user_count": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Also request Discord's subscriber count for each event"
                    ),
                },
            },
            "required": ["server_id"],
        },
    ),
    Tool(
        name="edit_scheduled_event",
        description=(
            "Edit a guild scheduled event (dry-run by default); at least one "
            "field is required. Discord decides permission (Manage Events) and "
            "rejects entity/channel combinations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "event_id": {"type": "string", "description": "Scheduled event ID"},
                "name": {"type": "string", "description": "New event name"},
                "description": {
                    "type": "string",
                    "description": "New event description",
                },
                "channel_id": {
                    "type": "string",
                    "description": "New stage/voice channel for the event",
                },
                "start_time": {
                    "type": "string",
                    "description": "New ISO-8601 start time with offset",
                },
                "end_time": {
                    "type": "string",
                    "description": "New ISO-8601 end time with offset",
                },
                "privacy_level": {
                    "type": "string",
                    "enum": ["guild_only"],
                    "description": "New privacy level",
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["stage_instance", "voice_channel", "external"],
                    "description": "New entity type for the event",
                },
                "image_url": {
                    "type": "string",
                    "description": "New http(s) URL for the event cover image",
                },
                "location": {
                    "type": "string",
                    "description": "New physical location (external events)",
                },
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for the edit",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without editing; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "event_id"],
        },
    ),
    Tool(
        name="delete_scheduled_event",
        description=(
            "Delete a guild scheduled event permanently (dry-run by default; "
            "reason required). Discord decides permission (Manage Events); the "
            "event and its RSVPs cannot be recovered."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "event_id": {"type": "string", "description": "Scheduled event ID"},
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for the deletion",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without deleting; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "event_id", "reason"],
        },
    ),
    Tool(
        name="start_scheduled_event",
        description=(
            "Start a scheduled event early, moving it to active (dry-run by "
            "default). Discord decides permission (Manage Events) and the "
            "scheduled -> active state transition."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "event_id": {"type": "string", "description": "Scheduled event ID"},
                "reason": {
                    "type": "string",
                    "description": "Optional audit-log reason for the status change",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without starting; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "event_id"],
        },
    ),
    Tool(
        name="end_scheduled_event",
        description=(
            "End a running scheduled event, moving it to completed (dry-run by "
            "default). Discord decides permission (Manage Events) and the "
            "active -> completed state transition."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "event_id": {"type": "string", "description": "Scheduled event ID"},
                "reason": {
                    "type": "string",
                    "description": "Optional audit-log reason for the status change",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without ending; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "event_id"],
        },
    ),
    Tool(
        name="cancel_scheduled_event",
        description=(
            "Cancel a guild scheduled event (dry-run by default; reason "
            "required). Discord decides permission (Manage Events); a canceled "
            "event cannot be restarted - create a new one."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "event_id": {"type": "string", "description": "Scheduled event ID"},
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for the cancellation",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without cancelling; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "event_id", "reason"],
        },
    ),
    Tool(
        name="list_scheduled_event_users",
        description=(
            "List users who RSVP'd for a scheduled event. No confirmation gate; "
            "the members intent is required for member profiles beyond the bot "
            "itself. limit caps how many users are fetched (default 100, max 1000)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "event_id": {"type": "string", "description": "Scheduled event ID"},
                "limit": {
                    "type": "integer",
                    "default": 100,
                    "description": "Maximum users to fetch (1-1000)",
                },
            },
            "required": ["server_id", "event_id"],
        },
    ),
    Tool(
        name="create_stage_instance",
        description=(
            "Open a stage instance in a stage channel (dry-run by default). "
            "channel_id must be a stage channel; Discord decides permission "
            "(Manage Channels; Mention Everyone for send_start_notification)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {"type": "string", "description": "Stage channel ID"},
                "topic": {"type": "string", "description": "Stage topic"},
                "privacy_level": {
                    "type": "string",
                    "enum": ["guild_only"],
                    "description": "Stage privacy level (defaults to guild_only)",
                },
                "send_start_notification": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Push a start notification to @everyone (needs Mention "
                        "Everyone)"
                    ),
                },
                "scheduled_event_id": {
                    "type": "string",
                    "description": "Optional guild scheduled event to link to the stage",
                },
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for the creation",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without opening; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "channel_id", "topic"],
        },
    ),
    Tool(
        name="get_stage_instance",
        description=(
            "Fetch the stage instance currently running in a stage channel. No "
            "confirmation gate; raises a clear error when no instance is running. "
            "Discord decides visibility."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {"type": "string", "description": "Stage channel ID"},
            },
            "required": ["server_id", "channel_id"],
        },
    ),
    Tool(
        name="edit_stage_instance",
        description=(
            "Edit the running stage instance's topic or privacy level (dry-run "
            "by default); at least one field is required. Discord decides "
            "permission (Manage Channels)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {"type": "string", "description": "Stage channel ID"},
                "topic": {"type": "string", "description": "New stage topic"},
                "privacy_level": {
                    "type": "string",
                    "enum": ["guild_only"],
                    "description": "New privacy level",
                },
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for the edit",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without editing; returns a confirmToken",
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
        name="delete_stage_instance",
        description=(
            "Close the running stage instance (dry-run by default; reason "
            "required). Discord decides permission (Manage Channels); the "
            "instance cannot be reopened, only recreated."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Discord server ID"},
                "channel_id": {"type": "string", "description": "Stage channel ID"},
                "reason": {
                    "type": "string",
                    "description": "Audit-log reason for closing the stage",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preview without closing; returns a confirmToken",
                },
                "confirm_token": {
                    "type": "string",
                    "description": "Token from the dry run; required when dry_run is false",
                },
            },
            "required": ["server_id", "channel_id", "reason"],
        },
    ),
]
