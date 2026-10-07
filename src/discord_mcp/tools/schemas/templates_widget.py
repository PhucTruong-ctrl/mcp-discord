from mcp.types import Tool


TEMPLATES_WIDGET_TOOLS = [
    Tool(
        name="list_templates",
        description=(
            "List the guild's account-level server templates (dry-run free, read only). "
            "Requires MANAGE_GUILD; Discord decides permission."
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
        name="create_template",
        description=(
            "Create a server-level template (dry-run by default). Requires "
            "MANAGE_GUILD, which Discord enforces. discord.py 2.7.1's "
            "Guild.create_template takes no audit-log reason, so 'reason' is "
            "validated and echoed in the result but never transmitted."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "name": {"type": "string", "description": "Template name"},
                "description": {
                    "type": "string",
                    "description": "Template description",
                },
                "reason": {
                    "type": "string",
                    "description": "Operator reason (not transmitted)",
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
            "required": ["server_id", "name"],
        },
    ),
    Tool(
        name="get_template",
        description=(
            "Fetch an account-level template by raw code or a "
            "discord.gg/<code> / discord.com/invite/<code> URL. Read only; "
            "templates are account-level, so the Discord client answers, not "
            "the cached guild."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Template code or invite-style URL",
                },
            },
            "required": ["code"],
        },
    ),
    Tool(
        name="sync_template",
        description=(
            "Sync a template to its source guild's current state (dry-run by "
            "default), then optionally apply name/description through "
            "Template.edit. Requires MANAGE_GUILD in the source guild. "
            "discord.py takes MISSING for unset fields, so omitted values are "
            "left untouched."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Template code or invite-style URL",
                },
                "name": {
                    "type": "string",
                    "description": "Name applied after the sync",
                },
                "description": {
                    "type": "string",
                    "description": "Description applied after the sync",
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
            "required": ["code"],
        },
    ),
    Tool(
        name="edit_template",
        description=(
            "Edit a template's name/description (dry-run by default). Requires "
            "MANAGE_GUILD in the source guild. At least one of name or "
            "description must be supplied; an empty edit is rejected."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Template code or invite-style URL",
                },
                "name": {"type": "string", "description": "New template name"},
                "description": {
                    "type": "string",
                    "description": "New template description",
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
            "required": ["code"],
        },
    ),
    Tool(
        name="delete_template",
        description=(
            "Permanently delete a template (dry-run by default); reason is "
            "required. Irreversible — Discord has no undelete. "
            "discord.py 2.7.1's Template.delete() takes no audit-log reason, "
            "so the reason is validated for the gate but never transmitted."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Template code or invite-style URL",
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
            "required": ["code", "reason"],
        },
    ),
    Tool(
        name="get_guild_preview",
        description=(
            "Fetch a guild's public preview (banner assets, features, "
            "approximate member/presence counts) through the Discord client. "
            "Read only; Discord answers 404 for a guild without a preview."
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
        name="get_widget_settings",
        description=(
            "Read the guild widget settings. 'enabled' comes from "
            "Guild.widget_enabled because Widget carries no such flag, and "
            "Discord refuses the widget payload while the widget is off — in "
            "that case inviteUrl/jsonUrl/presenceCount are null. "
            "'channelId' is the configured widget channel, which the widget "
            "payload does not expose."
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
        name="edit_widget_settings",
        description=(
            "Enable/disable the guild widget or set its channel (dry-run by "
            "default). Requires MANAGE_GUILD. At least one of enabled or "
            "channel_id is required; unset fields are omitted so discord.py "
            "keeps them (MISSING, not null). The channel is resolved and "
            "passed as an object because Discord rejects a bare id."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID"},
                "enabled": {
                    "type": "boolean",
                    "description": "Whether the widget is enabled",
                },
                "channel_id": {
                    "type": "string",
                    "description": "Widget channel ID (must exist in the guild)",
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
            "required": ["server_id"],
        },
    ),
]
