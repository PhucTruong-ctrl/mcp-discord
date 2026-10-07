# Destructive Actions Safety Policy

## Purpose
This policy defines guardrails for destructive or high-blast Discord operations exposed by this MCP server.

## Core controls
For destructive operations, handlers must support a two-step execution model:
1. **Dry run** (`dry_run=true`) returns impact details and a confirm token.
2. **Execute path** (`dry_run=false`) requires valid `confirm_token` when the tool policy marks confirmation as mandatory.

Supporting details:
- `reason` is required for destructive moderation/policy actions.
- `DISCORD_MCP_CONFIRM_SECRET` must be configured when confirmation is required.
- Missing/invalid confirm token must fail fast with explicit errors.

## Confirm-token contract
- Token generation/verification is deterministic via shared safety helpers.
- Dry-run payload includes `confirmToken` from `build_dry_run_result(action, targets, details)`.
- Execute path validates token with `verify_confirm_token(action, targets, confirm_token)`.
- **Reason-binding nuance**: destructive tools include `reason` inside the confirm-token `targets`, so a token issued for one `reason` cannot be reused for the same targets under a different reason.

## Tools that explicitly require `confirm_token` on execute path
When `dry_run=false`, 72 tools require a valid token. Source: each tool's `inputSchema` declares `dry_run` in `compose_tool_registry()`.

### Moderation
`moderation_bulk_delete`, `moderation_ban_member`, `moderation_kick_member`, `moderation_timeout_member`

### Incident
`incident_apply_lockdown`, `incident_rollback_lockdown`

### AutoMod
`automod_apply_ruleset`, `automod_rollback_ruleset`

### Roles
`add_roles_bulk`, `remove_roles_bulk`, `set_member_roles`, `reorder_roles`, `set_role_icon`

### Expansion utilities
`bulk_ban_members`, `prune_inactive_members`, `delete_category`

### Threads and channels
`create_thread`, `add_thread_member`, `remove_thread_member`, `edit_thread`, `delete_thread`, `create_thread_from_message`, `clone_channel`, `create_announcement_channel`, `create_stage_channel`, `follow_channel`, `sync_channel_permissions`, `set_voice_channel_status`, `pin_message`, `unpin_message`, `clear_message_reactions`

### Members, voice and profile
`change_member_voice_state`, `move_member_voice`, `request_to_speak`, `edit_member_profile`, `update_bot_profile`

### Emoji, sticker and soundboard
`create_emoji`, `edit_emoji`, `delete_emoji`, `create_application_emoji`, `edit_application_emoji`, `delete_application_emoji`, `create_sticker`, `edit_sticker`, `delete_sticker`, `create_soundboard_sound`, `edit_soundboard_sound`, `delete_soundboard_sound`

### Webhooks
`edit_webhook`, `delete_webhook`, `edit_webhook_message`, `delete_webhook_message`

### Scheduled events and stage
`create_scheduled_event`, `edit_scheduled_event`, `delete_scheduled_event`, `start_scheduled_event`, `end_scheduled_event`, `cancel_scheduled_event`, `create_stage_instance`, `edit_stage_instance`, `delete_stage_instance`

### Monetization, templates and widget
`create_entitlement`, `consume_entitlement`, `delete_entitlement`, `sync_app_commands`, `create_template`, `sync_template`, `edit_template`, `delete_template`, `edit_widget_settings`

### Invites
`create_invite`, `delete_invite`

## Operational guidance
- Always run destructive tools in dry-run mode first.
- Surface dry-run output to operators before execute confirmation.
- Reject direct execute requests that skip required token or reason.
- Keep policy behavior consistent across schema definitions and handler validation logic.
