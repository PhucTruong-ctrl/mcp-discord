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
- Dry-run payload includes `confirmToken` (or `confirm_token` in tool-specific payloads where implemented).
- Execute path validates token against the same action/targets or action+reason tuple used by the handler.

## Tools that explicitly require `confirm_token` on execute path

When `dry_run=false`, these tools require a valid token in this codebase:

1. `moderation_bulk_delete`
2. `moderation_timeout_member`
3. `moderation_kick_member`
4. `moderation_ban_member`
5. `incident_apply_lockdown`
6. `incident_rollback_lockdown`
7. `automod_apply_ruleset`
8. `automod_rollback_ruleset`

## Operational guidance

- Always run destructive tools in dry-run mode first.
- Surface dry-run output to operators before execute confirmation.
- Reject direct execute requests that skip required token or reason.
- Keep policy behavior consistent across schema definitions and handler validation logic.
