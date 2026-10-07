from mcp.types import Tool


MONETIZATION_APPCMDS_TOOLS = [
    Tool(
        name="list_skus",
        description=(
            "List the application's SKUs (id, name, slug, type). Only a monetized "
            "application has SKUs: an empty list means this application cannot sell "
            "anything, which the payload reports as monetized=false rather than an "
            "error. Read-only; no confirmation gate."
        ),
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="list_entitlements",
        description=(
            "List entitlements granted for this application, filterable by SKU, "
            "user, guild and deletion/end state. Only a monetized application has "
            "entitlements. Read-only; no confirmation gate."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum entitlements to return (default 100)",
                },
                "sku_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only entitlements for these SKU IDs",
                },
                "user_id": {
                    "type": "string",
                    "description": "Only entitlements owned by this user",
                },
                "guild_id": {
                    "type": "string",
                    "description": "Only entitlements owned by this guild",
                },
                "exclude_ended": {
                    "type": "boolean",
                    "default": False,
                    "description": "Skip entitlements whose end time has passed",
                },
                "exclude_deleted": {
                    "type": "boolean",
                    "default": True,
                    "description": "Skip deleted entitlements",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="get_entitlement",
        description=(
            "Fetch one entitlement by ID. Raises an error naming the ID when Discord "
            "reports it unknown. Read-only; no confirmation gate."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entitlement_id": {"type": "string", "description": "Entitlement ID"},
            },
            "required": ["entitlement_id"],
        },
    ),
    Tool(
        name="create_entitlement",
        description=(
            "Create a test entitlement granting a SKU to a user or guild (dry-run by "
            "default). owner_type must be 'user' or 'guild'. reason is required for "
            "the confirm gate; discord.py's create_entitlement sends no audit-log "
            "reason to Discord. Only works for a monetized application."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "sku_id": {"type": "string", "description": "SKU ID to grant"},
                "owner_id": {"type": "string", "description": "User or guild ID"},
                "owner_type": {
                    "type": "string",
                    "enum": ["user", "guild"],
                    "description": "Whether owner_id is a user or a guild",
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
            "required": ["sku_id", "owner_id", "owner_type", "reason"],
        },
    ),
    Tool(
        name="consume_entitlement",
        description=(
            "Mark a one-time-purchase entitlement as consumed (dry-run by default; "
            "irreversible - Discord will not allow consuming it twice). reason is "
            "optional and not transmitted: Entitlement.consume() takes no audit-log "
            "reason. Discord decides whether the entitlement is consumable."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entitlement_id": {"type": "string", "description": "Entitlement ID"},
                "reason": {
                    "type": "string",
                    "description": (
                        "Optional reason; validated for the gate but not sent "
                        "(consume takes no audit-log reason)"
                    ),
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
            "required": ["entitlement_id"],
        },
    ),
    Tool(
        name="delete_entitlement",
        description=(
            "Delete an entitlement (dry-run by default; irreversible). reason is "
            "required for the confirm gate; discord.py's Entitlement.delete() sends "
            "no audit-log reason to Discord."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entitlement_id": {"type": "string", "description": "Entitlement ID"},
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
            "required": ["entitlement_id", "reason"],
        },
    ),
    Tool(
        name="list_app_commands",
        description=(
            "List the application's registered application commands. Omit guild_id "
            "for the global command scope; supply it to list one guild's commands. "
            "Read-only; no confirmation gate."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "guild_id": {
                    "type": "string",
                    "description": "Guild scope; omitted lists global commands",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="get_app_command",
        description=(
            "Fetch one registered application command by ID from the global scope, "
            "or from a guild when guild_id is given. Discord reports a command "
            "missing from the requested scope as unknown. Read-only; no "
            "confirmation gate."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "command_id": {"type": "string", "description": "Command ID"},
                "guild_id": {
                    "type": "string",
                    "description": "Guild scope; omitted reads the global scope",
                },
            },
            "required": ["command_id"],
        },
    ),
    Tool(
        name="sync_app_commands",
        description=(
            "Synchronise the local application-command tree to Discord (dry-run by "
            "default): registers, edits and deletes registered commands to match the "
            "local tree, so commands missing from the tree are removed. Omit "
            "guild_id for the global scope; supply it to sync one guild. Discord can "
            "take up to an hour to propagate global command changes. reason is "
            "optional and not transmitted: CommandTree.sync() takes no audit-log "
            "reason."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "guild_id": {
                    "type": "string",
                    "description": "Guild scope; omitted syncs the global tree",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Optional reason; validated for the gate but not sent "
                        "(sync takes no audit-log reason)"
                    ),
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
]
