## Review drift fix: expansion fillers taxonomy split

### What changed

The 15 expansion filler tools (92-106) had two distinct runtime behaviors that were inaccurately documented as "all synthetic":

**Tools 92-102 (synthetic-only):** `bulk_ban_members`, `prune_inactive_members`, `remove_member_timeout`, `unban_member`, `create_category`, `rename_category`, `move_category`, `delete_category`, `create_incident_room`, `append_incident_event`, `close_incident` — purely synthetic, no gateway interaction.

**Tools 103-106 (gateway-aware with synthetic fallback):** `list_auto_moderation_rules`, `create_auto_moderation_rule`, `update_auto_moderation_rule`, `automod_export_rules` — use Discord API via gateway when available; fall back to synthetic when absent.

### Files changed
1. `docs/product/tool-catalog.md` — Split the expansion fillers section into synthetic-only (92-102) and gateway-aware (103-106) with accurate descriptions.
2. `tests/test_tool_runtime_contracts.py` — Updated module docstring, class docstring, and test method name/docstring to accurately describe gateway-absent fallback semantics instead of claiming "never uses gateway". Added cross-reference to test_automod_runtime_tools.py.
3. `tests/test_automod_runtime_tools.py` — Added class docstring explaining gateway-aware semantics with cross-reference to contract tests.

### Verification
- `uv run pytest -q`: 196 passed, 34 subtests passed
- `uv run python -m compileall src/discord_mcp`: clean (no errors)
