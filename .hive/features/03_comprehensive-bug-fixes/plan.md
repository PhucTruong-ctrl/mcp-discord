## Discovery

**Q: What exactly are the bugs identified?**
A:
1. `Enum` Bug: Calling `gateway.fetch_audit_entries` with `action_type=None` passed `action=None` to `discord.py`'s `guild.audit_logs`. `discord.py` internally checks `action.value` if it's not `MISSING`, crashing because `None` is not `MISSING`. Same issue occurred in `create_role` passing `permissions=None`.
2. Forum post tools crash because `resolve_forum_post` wasn't defined in `DiscordGateway`.
3. `automod_get_ruleset` crashed because `ruleset.rules` missing from input failed the `isinstance(list)` check.

**Research:**
- Fixed `fetch_audit_entries` to only pass `action` when explicitly set.
- Fixed `role_governance.py` (`create_role` and `update_role`) to correctly typecast permissions to `discord.Permissions` and skip passing `None` down to the `discord.py` APIs.
- Added `resolve_forum_post` to `DiscordGateway` (aliasing to `resolve_thread`).
- Relaxed the shape check in `automod_policy.py` to default `.get("rules")` to `[]` before validation, to support inputs that lack the `rules` array.

## Non-Goals
- We are not restructuring other tools beyond these fixes.
- We are not redesigning the API or registry, we just ensure these specific handler bugs are fixed.

## Tasks

### 1. Identify and fix Enum 'value' bug
**Depends on**: none
**Files**:
- Modify: `src/discord_mcp/services/discord_gateway.py:215-230`
- Modify: `src/discord_mcp/tools/handlers/role_governance.py`
**What**: Stop passing `None` as optional arguments to discord.py when it expects `MISSING`. Pass kwargs dynamically. Ensure types match what discord.py requires (e.g. `discord.Permissions(int(...))`).
**Must NOT**: Do not change function signatures in tools.
**Verify**: `uv run pytest tests/test_audit_analytics_tools.py tests/test_role_governance_tools.py`

### 2. Implement resolve_forum_post
**Depends on**: none
**Files**:
- Modify: `src/discord_mcp/services/discord_gateway.py`
**What**: Add `resolve_forum_post(post_id, server_id)` to `DiscordGateway` which calls `resolve_thread`.
**Must NOT**: Do not change return type.
**Verify**: `uv run pytest tests/test_forum_intel_tools.py`

### 3. Fix automod_get_ruleset rules array bug
**Depends on**: none
**Files**:
- Modify: `src/discord_mcp/tools/handlers/automod_policy.py`
**What**: In `_validate_ruleset_shape`, handle the missing `rules` field gracefully by assigning `[]` or allowing it if it's missing, avoiding the `NoneType` crash on `isinstance()`.
**Must NOT**: Do not remove validation for `name` and basic dict structure.
**Verify**: `uv run pytest tests/test_automod_policy_tools.py`