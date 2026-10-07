from mcp.types import Tool


WEBHOOK_MGMT_TOOLS = [
    Tool(
        name="get_webhook",
        description=(
            "Fetch a webhook's details by ID using its token (no confirmation "
            "gate). The token is used only for the authenticated lookup and is "
            "never echoed back: the payload carries a masked token (last 4 "
            "characters) and a masked URL."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "webhook_id": {"type": "string", "description": "Webhook ID"},
                "webhook_token": {
                    "type": "string",
                    "description": (
                        "Webhook token; used for the lookup, never returned"
                    ),
                },
            },
            "required": ["webhook_id", "webhook_token"],
        },
    ),
    Tool(
        name="edit_webhook",
        description=(
            "Edit a webhook's name, target channel, or avatar (dry-run by "
            "default). 'avatar_url' must be an http(s) image URL; it is "
            "downloaded and sent as bytes. Moving the webhook to another "
            "channel uses the authenticated (bot-token) endpoint, so the bot "
            "must have access to the target channel; Discord decides "
            "permission. The webhook token is never echoed back."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "webhook_id": {"type": "string", "description": "Webhook ID"},
                "webhook_token": {
                    "type": "string",
                    "description": (
                        "Webhook token; used for the call, never returned"
                    ),
                },
                "name": {"type": "string", "description": "New default name"},
                "channel_id": {
                    "type": "string",
                    "description": "New channel ID for the webhook",
                },
                "avatar_url": {
                    "type": "string",
                    "description": "New avatar as an http(s) image URL",
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
            "required": ["webhook_id", "webhook_token"],
        },
    ),
    Tool(
        name="delete_webhook",
        description=(
            "Delete a webhook permanently (dry-run by default; irreversible). "
            "Requires the webhook's token; 'reason' is required and recorded "
            "in the audit log. Discord decides permission."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "webhook_id": {"type": "string", "description": "Webhook ID"},
                "webhook_token": {
                    "type": "string",
                    "description": "Webhook token; never returned",
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
            "required": ["webhook_id", "webhook_token", "reason"],
        },
    ),
    Tool(
        name="get_webhook_message",
        description=(
            "Fetch a message previously sent by a webhook (no confirmation "
            "gate). Requires the webhook's token; embeds and attachments are "
            "returned inline. Discord decides visibility."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "webhook_id": {"type": "string", "description": "Webhook ID"},
                "webhook_token": {
                    "type": "string",
                    "description": "Webhook token; never returned",
                },
                "message_id": {
                    "type": "string",
                    "description": "Message ID to fetch",
                },
            },
            "required": ["webhook_id", "webhook_token", "message_id"],
        },
    ),
    Tool(
        name="edit_webhook_message",
        description=(
            "Edit the content of a message owned by a webhook (dry-run by "
            "default). 'content' is required — an empty edit is a no-op the "
            "caller should not make. Only the webhook's own messages can be "
            "edited; Discord decides permission. The token is never echoed "
            "back."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "webhook_id": {"type": "string", "description": "Webhook ID"},
                "webhook_token": {
                    "type": "string",
                    "description": "Webhook token; never returned",
                },
                "message_id": {
                    "type": "string",
                    "description": "Message ID to edit",
                },
                "content": {
                    "type": "string",
                    "description": "New message content",
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
            "required": ["webhook_id", "webhook_token", "message_id"],
        },
    ),
    Tool(
        name="delete_webhook_message",
        description=(
            "Delete a message owned by a webhook (dry-run by default; "
            "irreversible). 'reason' is required by the confirmation gate; "
            "Discord's webhook-message delete endpoint accepts no audit-log "
            "reason, so it is validated but not transmitted. Discord decides "
            "permission."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "webhook_id": {"type": "string", "description": "Webhook ID"},
                "webhook_token": {
                    "type": "string",
                    "description": "Webhook token; never returned",
                },
                "message_id": {
                    "type": "string",
                    "description": "Message ID to delete",
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
            "required": ["webhook_id", "webhook_token", "message_id", "reason"],
        },
    ),
]
