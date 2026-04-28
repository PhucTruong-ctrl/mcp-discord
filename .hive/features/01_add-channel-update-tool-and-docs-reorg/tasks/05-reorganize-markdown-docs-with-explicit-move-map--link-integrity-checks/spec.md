# Task: 05-reorganize-markdown-docs-with-explicit-move-map--link-integrity-checks

## Feature: add-channel-update-tool-and-docs-reorg

## Dependencies

- **4. update-docs-for-channel-crudadmin-tools** (04-update-docs-for-channel-crudadmin-tools)

## Plan Section

### 5. Reorganize markdown docs with explicit move map + link integrity checks

**Depends on**: 4

**Files:**
- Create: `docs/README.md`
- Move/Modify:
  - `docs/tool-catalog.md` → `docs/product/tool-catalog.md`
  - `docs/waves/01-10-rollout.md` → `docs/product/rollout/01-10-rollout.md`
  - `docs/safety/destructive-actions-policy.md` → `docs/product/safety/destructive-actions-policy.md`
  - `docs/OPENCLAW_ECOSYSTEM_RESEARCH.md` → `docs/product/research/openclaw-ecosystem-research.md`
  - `README_GAP_ANALYSIS.md` → `docs/analysis/README_GAP_ANALYSIS.md`
  - `ALL_30_TOOLS_REFERENCE.md` → `docs/analysis/ALL_30_TOOLS_REFERENCE.md`
  - `ANALYSIS_INDEX.md` → `docs/analysis/ANALYSIS_INDEX.md`
  - `PHASE1_IMPLEMENTATION.md` → `docs/analysis/PHASE1_IMPLEMENTATION.md`
  - `TOOL_GAP_ANALYSIS.md` → `docs/analysis/TOOL_GAP_ANALYSIS.md`
  - `RESEARCH_INDEX.md` → `docs/analysis/RESEARCH_INDEX.md`
  - `CAPABILITY_MATRIX.md` → `docs/analysis/CAPABILITY_MATRIX.md`
  - `ECOSYSTEM_SUMMARY.md` → `docs/analysis/ECOSYSTEM_SUMMARY.md`
  - `docs/plans/*` → `docs/meta/plans/*`
  - `docs/agent-sessions/*` → `docs/meta/agent-sessions/*`
- Modify: `README.md` and any moved md files containing internal links.

**What to do**:
- Step 1: Apply exact move map above.
- Step 2: Create `docs/README.md` index linking product/analysis/meta sections.
- Step 3: Update known links from discovery:
  - `README.md` docs pointers
  - ecosystem references to moved research file
- Step 4: Run markdown link integrity check script and fix all broken local links.

**Must NOT do**:
- Must not move `.pytest_cache/README.md`.
- Must not keep stale paths to pre-move locations.

**References**:
- `README.md:18-20`
- `CAPABILITY_MATRIX.md:279`
- `ECOSYSTEM_SUMMARY.md:7,172`

**Verify**:
- [ ] Run: `python - <<'PY'
from pathlib import Path
import re
root = Path('.')
mds = [p for p in root.rglob('*.md') if '.pytest_cache' not in p.parts]
pat = re.compile(r'\[[^\]]+\]\(([^)]+)\)')
bad=[]
for f in mds:
    txt=f.read_text(encoding='utf-8', errors='ignore')
    for m in pat.finditer(txt):
        u=m.group(1).strip()
        if not u or u.startswith(('http://','https://','#','mailto:')):
            continue
        rel=u.split('#',1)[0]
        t=(f.parent/rel).resolve()
        if not t.exists():
            bad.append((str(f),u))
if bad:
    print('BROKEN', len(bad))
    for x in bad[:200]:
        print(x[0], '->', x[1])
    raise SystemExit(1)
print('OK', len(mds), 'markdown files checked')
PY` → exits 0, prints `OK ...`
- [ ] Run: `pytest -q` → pass

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
- 03-implement-textvoiceforum-createupdate-handlers-with-explicit-fallback-behavior: Implemented create/update handlers for text, voice, and forum channels in src/discord_mcp/tools/handlers/channels.py with partial-update semantics and explicit errors for unsupported fields. Added the new channel tools to the schema and router, plus contract tests covering creation, update behavior, and fallback/error cases. Verified with `python -m unittest tests.test_channel_admin_tools tests.test_inventory_tools tests.test_topology_tools -q` (26 tests passed).
- 04-update-docs-for-channel-crudadmin-tools: Updated docs/tool-catalog.md and README.md to document the channel CRUD/admin surface, including create/read/update/delete mapping, field contracts, and forum text fallback behavior. Verified the requested entrypoint wiring tests with a stubbed runtime; the repo’s direct pytest invocation is unavailable in this environment because pytest is not installed.
