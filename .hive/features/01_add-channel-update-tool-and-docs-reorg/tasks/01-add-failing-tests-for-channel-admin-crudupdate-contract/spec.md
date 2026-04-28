# Task: 01-add-failing-tests-for-channel-admin-crudupdate-contract

## Feature: add-channel-update-tool-and-docs-reorg

## Dependencies

_None_

## Plan Section

### 1. Add failing tests for channel admin CRUD/update contract

**Depends on**: none

**Files:**
- Create: `tests/test_channel_admin_tools.py`
- Modify: `tests/test_entrypoint_wiring.py`

**What to do**:
- Step 1: Add registry/router presence tests for:
  - `create_voice_channel`, `create_forum_channel`
  - `update_text_channel`, `update_voice_channel`, `update_forum_channel`
- Step 2: Add async tests for each update tool (field-by-field positive cases).
- Step 3: Add negative tests:
  - wrong channel type
  - missing required identifiers
  - unknown fields produce `unsupported_fields`
  - forum unsupported-by-library field produces `field_not_supported_by_library`
- Step 4: Add read-coverage assertion tests proving existing read tools return enough fields for text/voice/forum admin workflows.
- Step 5: Run tests to confirm initial FAIL.
  - Run: `pytest tests/test_channel_admin_tools.py -q`
  - Expected: FAIL due to missing implementation.

**Must NOT do**:
- Must not implement production handlers in this task.

**References**:
- `tests/test_inventory_tools.py:67-204`
- `tests/test_entrypoint_wiring.py:55-83`

**Verify**:
- [ ] Run: `pytest tests/test_channel_admin_tools.py -q` → fails for missing implementation

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
