from mcp.types import Tool


THREAD_MANAGEMENT_TOOLS = [
    Tool(
        name="create_thread",
        description=(
            "Create a thread in a text channel (dry-run by default). 'type' selects "
            "public_thread or private_thread; omitted means discord.py's default "
            "(private thread). Forum/media channels are rejected — use the forum-post "
            "tools. Discord decides permission (create_public_threads / "
            "create_private_threads)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "channel_id": {
                    "type": "string",
                    "description": "Parent text channel ID",
                },
                "name": {"type": "string", "description": "Thread name"},
                "auto_archive_duration": {
                    "type": "integer",
                    "enum": [60, 1440, 4320, 10080],
                    "description": "Minutes of inactivity before the thread auto-archives",
                },
                "slowmode_delay": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 21600,
                    "description": "Slowmode delay in seconds (0-21600)",
                },
                "type": {
                    "type": "string",
                    "enum": ["public_thread", "private_thread"],
                    "description": (
                        "Thread type; omitted creates a private thread "
                        "(discord.py 2.7.1 default)"
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
            "required": ["server_id", "channel_id", "name"],
        },
    ),
    Tool(
        name="join_thread",
        description=(
            "Join a thread as the bot. No confirmation gate; Discord decides whether "
            "the bot may see and join the thread."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "thread_id": {"type": "string", "description": "Thread ID"},
            },
            "required": ["server_id", "thread_id"],
        },
    ),
    Tool(
        name="leave_thread",
        description=(
            "Leave a thread the bot has joined. No confirmation gate; only affects "
            "the bot's own membership."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "thread_id": {"type": "string", "description": "Thread ID"},
            },
            "required": ["server_id", "thread_id"],
        },
    ),
    Tool(
        name="add_thread_member",
        description=(
            "Add a member to a thread (dry-run by default). Discord decides "
            "permission (only the thread owner/moderators may add members to private "
            "threads)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "thread_id": {"type": "string", "description": "Thread ID"},
                "user_id": {"type": "string", "description": "Member ID to add"},
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
            "required": ["server_id", "thread_id", "user_id"],
        },
    ),
    Tool(
        name="remove_thread_member",
        description=(
            "Remove a member from a thread (dry-run by default). Discord decides "
            "permission (manage_threads, or the member removing themselves)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "thread_id": {"type": "string", "description": "Thread ID"},
                "user_id": {"type": "string", "description": "Member ID to remove"},
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
            "required": ["server_id", "thread_id", "user_id"],
        },
    ),
    Tool(
        name="edit_thread",
        description=(
            "Edit thread properties (dry-run by default); at least one field is "
            "required. The thread must be unarchived before it can be edited, "
            "'pinned' only affects forum posts, and 'invitable' only applies to "
            "private threads — Discord enforces these rules."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "thread_id": {"type": "string", "description": "Thread ID"},
                "name": {"type": "string", "description": "New thread name"},
                "archived": {
                    "type": "boolean",
                    "description": "Archive or unarchive the thread",
                },
                "locked": {
                    "type": "boolean",
                    "description": "Lock or unlock the thread",
                },
                "invitable": {
                    "type": "boolean",
                    "description": "Private threads only: who may add members",
                },
                "pinned": {
                    "type": "boolean",
                    "description": "Pin or unpin (forum posts only)",
                },
                "slowmode_delay": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 21600,
                    "description": "Slowmode delay in seconds (0-21600)",
                },
                "auto_archive_duration": {
                    "type": "integer",
                    "enum": [60, 1440, 4320, 10080],
                    "description": "Minutes of inactivity before the thread auto-archives",
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
            "required": ["server_id", "thread_id"],
        },
    ),
    Tool(
        name="delete_thread",
        description=(
            "Permanently delete a thread (dry-run by default); reason is required "
            "and Discord enforces manage_threads. Irreversible — the thread and its "
            "messages cannot be recovered."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "thread_id": {"type": "string", "description": "Thread ID"},
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
            "required": ["server_id", "thread_id", "reason"],
        },
    ),
    Tool(
        name="list_active_threads",
        description=(
            "List active (non-archived) threads in a server, including private "
            "threads the bot can see; Discord decides visibility."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
            },
            "required": ["server_id"],
        },
    ),
]
