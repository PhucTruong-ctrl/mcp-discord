# Task: 03-implement-textvoiceforum-createupdate-handlers-with-explicit-fallback-behavior

## Feature: add-channel-update-tool-and-docs-reorg

## Dependencies

- **2. extend-schemas-and-router-wiring-for-new-channel-tools** (02-extend-schemas-and-router-wiring-for-new-channel-tools)

## Plan Section

### 3. Implement text/voice/forum create+update handlers with explicit fallback behavior

**Depends on**: 2

**Files:**
- Modify: `src/discord_mcp/tools/handlers/channels.py`
- Modify: `src/discord_mcp/services/discord_gateway.py` (if helper extraction needed)

**What to do**:
- Step 1: Implement `handle_create_voice_channel`, `handle_create_forum_channel`.
- Step 2: Implement `handle_update_text_channel`, `handle_update_voice_channel`, `handle_update_forum_channel`.
- Step 3: Enforce partial update semantics (only provided fields mutate).
- Step 4: Enforce strict field filtering and emit explicit errors for unsupported fields.
- Step 5: For forum: detect library support before applying each forum-specific field; if unsupported, return `field_not_supported_by_library`.
- Step 6: Run targeted tests until green.
  - Run: `pytest tests/test_channel_admin_tools.py -q`
  - Expected: PASS.

**Must NOT do**:
- Must not silently ignore unsupported fields.
- Must not regress `create_text_channel` / `delete_channel`.

**References**:
- `src/discord_mcp/tools/handlers/channels.py:6-56`
- `src/discord_mcp/services/discord_gateway.py:77-173`
- `src/discord_mcp/tools/handlers/forums.py:184-253`
- `src/discord_mcp/core/resolve.py:4-12`

**Verify**:
- [ ] Run: `pytest tests/test_channel_admin_tools.py tests/test_inventory_tools.py tests/test_topology_tools.py -q` → pass

---

## Task Type

modification

## Context

## discovery-findings

Discovery findings for feature add-channel-update-tool-and-docs-reorg:

User constraints:
- Need CRUD/update capability per channel type (text/voice/forum) with type-specific fields.
- Move markdown docs into docs/.
- Refactor existing tools is allowed (not strict non-breaking).

Codebase findings:
- Existing channel tools: create_text_channel, delete_channel only in src/discord_mcp/tools/schemas/channels.py and src/discord_mcp/tools/handlers/channels.py.
- Category operations exist in src/discord_mcp/tools/handlers/expansion_fillers.py and are routed in src/discord_mcp/tools/handlers/router.py.
- No existing direct update/edit channel tool for existing channels (topic/description/name/etc).
- Schema registry assembled in src/discord_mcp/tools/schemas/__init__.py and exported through composition helpers.

Docs findings:
- Many markdown docs currently in repo root (analysis/research docs) plus existing docs/* tree.
- Candidate migration includes moving root analysis docs into docs/analysis and normalizing product/meta docs under docs/product and docs/meta.
- README currently links to docs/tool-catalog.md, docs/waves/01-10-rollout.md, docs/safety/destructive-actions-policy.md and will require link updates after moves.
- Cross-doc links to docs/OPENCLAW_ECOSYSTEM_RESEARCH.md exist in CAPABILITY_MATRIX.md and ECOSYSTEM_SUMMARY.md and need updates if moved.

Testing findings:
- Common pattern: registry+router presence checks + async handler tests using fake gateway objects and JSON payload assertions.
- Relevant examples: tests/test_topology_tools.py, tests/test_inventory_tools.py, tests/test_moderation_core_tools.py, tests/test_entrypoint_wiring.py.
- Recommended to add new tests for update/edit tool(s) with per-channel-type validation and type-specific behavior checks.

---

## gateway-capability-findings

Additional gateway capability findings:
- src/discord_mcp/services/discord_gateway.py currently provides resolve/fetch helpers but no channel create/edit wrapper methods.
- Existing mutable channel-related operations are limited: create_text_channel + delete_channel, forum thread tag/archive edits, and category placeholder ops.
- No direct tool to edit existing text/voice/forum channel settings.
- Validation patterns to reuse: try_int/normalize_name helpers, typed channel resolution, forum tag membership checks.
- Safety patterns: dry-run + confirm token used for destructive ops.
- Recommended design: add dedicated create/update tools for text/voice/forum channels with type-specific fields and shared validation.

---

## learnings

Task 2 wiring touched schema/router only. Verification in this worktree required stubbing external runtime deps (discord, mcp, aiohttp) because direct unittest/pytest imports fail before test collection in the local environment; with stubs, compose_tool_registry exposes 103 unique tools and TOOL_ROUTER contains create_voice_channel, create_forum_channel, update_text_channel, update_voice_channel, and update_forum_channel.

## Completed Tasks

- 01-add-failing-tests-for-channel-admin-crudupdate-contract: Added contract tests for channel admin CRUD/update tooling in tests/test_channel_admin_tools.py and extended tests/test_entrypoint_wiring.py to snapshot the new registry/router surface. Verified with python -m unittest tests.test_channel_admin_tools -q; it fails as expected because the channel admin update implementation is still missing.
- 02-extend-schemas-and-router-wiring-for-new-channel-tools: Extended the channel schema registry and router wiring to expose create_voice_channel, create_forum_channel, and per-type update tools alongside the existing text channel tool. Kept business logic out of the router and preserved the schema registry composition order while adding the new channel tool entries. Verified registry/router wiring with a stubbed runtime, confirming 103 unique tools and router entries for all five new channel tool names; direct unittest invocation in this environment is blocked by missing external deps.
