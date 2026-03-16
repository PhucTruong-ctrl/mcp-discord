## Single-Guild Lock Design

Date: 2026-03-16
Topic: Enforce configured default Discord guild as the only visible/usable guild

### Goal

Ensure MCP behavior is constrained to the configured guild (`DEFAULT_GUILD_ID` or `DISCORD_GUILD_ID`) so users do not see or operate on other guilds the bot belongs to.

### Approved Behavior Contract

1. If default guild env is configured:
   - All server resolution is forced to that configured guild.
   - Any provided `server_id` is ignored silently.
   - `list_servers` returns exactly one entry (the configured guild).
   - If configured guild is inaccessible at runtime, server-scoped tools fail with a clear configuration/runtime error.

2. If default guild env is not configured:
   - Preserve current fallback behavior (existing multi-guild resolution semantics).

### Implementation Approach (Recommended)

Use a gateway-level lock so behavior is centralized and consistent across all handlers.

#### Files in scope

- `src/discord_mcp/services/discord_gateway.py`
  - Update `resolve_guild(server_id)`:
    - When default guild exists, resolve only configured default guild.
    - Ignore incoming `server_id` silently.
    - Raise canonical error if default guild is inaccessible.
  - Keep existing behavior when default guild is absent.

- `src/discord_mcp/tools/handlers/server_info.py`
  - `handle_list_servers`: resolve default guild via gateway and return only that server.
  - `handle_get_channels`: resolve via gateway (instead of direct `client.get_guild(int(server_id))`) so lock is always enforced.

### Error Handling

Canonical message for inaccessible configured guild:

`Configured default server '<id>' is not accessible by the bot. Check DISCORD_GUILD_ID/DEFAULT_GUILD_ID and bot guild membership.`

This should be produced consistently from gateway-backed server resolution paths.

### Testing and Verification

#### Unit tests

- `tests/test_gateway_unit.py`
  - default configured + valid guild -> resolves configured default regardless of passed `server_id`
  - default configured + missing guild -> raises canonical configuration/runtime error
  - no default configured -> existing behavior remains

#### Handler/contract tests

- `tests/test_tool_contracts_baseline.py` (and related contracts if needed)
  - `list_servers` returns exactly one server when default configured
  - `get_channels` ignores mismatched `server_id` and operates on locked default guild

#### Static checks

- Run `lsp_diagnostics` for touched files before completion claims.

### Non-goals

- No destructive moderation action changes.
- No public API redesign.
- No behavior change for deployments without a configured default guild.
