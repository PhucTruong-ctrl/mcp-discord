# Add channel update tools + docs reorganization

## Discovery

### Original Request
- "Hãy explore codebase hiện tại, đọc các file md, docs, xem những gì đã khả dụng... không thấy tool update channel topic/description trực tiếp cho kênh đã tồn tại... tao muốn mày làm thêm tool này, sau đó dọn dẹp, ogranize lại các file, di chuyển các file md liên quan docs vào docs"

### Interview Summary
- Scope channel mutation: implement **per-channel-type CRUD/update capability** (text, voice, forum) with type-specific fields.
- Forum scope: include guideline/tags-related settings where API supports; unsupported fields must fail explicitly.
- Docs scope: move docs markdown into `docs/` hierarchy.
- Compatibility: refactor allowed.

### Research Findings
- `src/discord_mcp/tools/schemas/channels.py:4-40`: currently only `create_text_channel` + `delete_channel`.
- `src/discord_mcp/tools/handlers/channels.py:6-56`: text create + generic delete handlers only.
- `src/discord_mcp/services/discord_gateway.py:10-250`: resolve/fetch helpers exist, no channel edit wrappers.
- `src/discord_mcp/tools/handlers/forums.py:184-253`: forum/thread post mutation exists, not forum-channel configuration CRUD.
- `tests/test_entrypoint_wiring.py:55-83`, `tests/test_inventory_tools.py:67-204`: registry/router + async fake gateway test style.
- `README.md:18-20`, `CAPABILITY_MATRIX.md:279`, `ECOSYSTEM_SUMMARY.md:7,172`: known links impacted by docs move.

### Supported Discord API fields (explicit contract)
| Channel type | Tool | Supported update fields | Unsupported behavior |
|---|---|---|---|
| Text | `update_text_channel` | `name`, `topic`, `nsfw`, `slowmode_delay`, `category_id`, `position` | return error with `unsupported_fields` when unknown fields are passed |
| Voice | `update_voice_channel` | `name`, `bitrate`, `user_limit`, `nsfw`, `category_id`, `position` | return error with `unsupported_fields` |
| Forum | `update_forum_channel` | `name`, `topic`/guidelines-equivalent supported by current lib, `available_tags`, `default_auto_archive_duration`, `slowmode_delay`, `category_id`, `position` | if library lacks a requested field, return explicit error (`field_not_supported_by_library`) |

### CRUD coverage statement (explicit)
- **Create**: `create_text_channel` (existing), plus new `create_voice_channel`, `create_forum_channel`.
- **Read**: existing inventory/read tools (`get_channels`, `get_channels_structured`, `get_channel_hierarchy`, `fetch_channel`-backed handlers). We will add test assertions proving these read APIs cover text/voice/forum admin workflows.
- **Update**: new per-type update tools.
- **Delete**: existing `delete_channel` (generic).

---

## Non-Goals (What we're NOT building)
- Full permission/RBAC simulation for channel mutations.
- Migrating unrelated non-doc markdown artifacts (e.g. `.pytest_cache/README.md`).
- Rewriting all historical analysis content quality; only structure + link integrity updates.
- Introducing a giant polymorphic single update tool.

---

## Ghost Diffs (Alternatives considered, rejected)
- **Single `update_channel` tool** → rejected; per-type tools give safer schemas and clearer UX.
- **Move docs without fixed map** → rejected; high breakage risk.
- **Keep root analysis docs indefinitely** → rejected; conflicts with user request to organize docs under `docs/`.

---

## Design Summary

We will add type-specific channel admin tools for text/voice/forum create+update, reuse existing delete_channel for delete, and explicitly verify existing read tooling closes CRUD for all three channel types.

Implementation layers:
- Schemas: `src/discord_mcp/tools/schemas/`
- Handlers: `src/discord_mcp/tools/handlers/`
- Router: `src/discord_mcp/tools/handlers/router.py`
- Registry composition: `src/discord_mcp/tools/schemas/__init__.py`

Docs will be reorganized with an explicit move map and a deterministic link integrity check.

---

## Tasks

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

### 2. Extend schemas and router wiring for new channel tools

**Depends on**: 1

**Files:**
- Modify: `src/discord_mcp/tools/schemas/channels.py`
- Modify: `src/discord_mcp/tools/schemas/__init__.py`
- Modify: `src/discord_mcp/tools/handlers/router.py`

**What to do**:
- Step 1: Add schema specs for new create/update tools.
- Step 2: Constrain input fields per channel type exactly per Supported Fields table.
- Step 3: Register schemas and route names to dedicated handlers.
- Step 4: Run wiring tests.
  - Run: `pytest tests/test_entrypoint_wiring.py -q`
  - Expected: PASS for wiring.

**Must NOT do**:
- Must not move business validation logic into router.

**References**:
- `src/discord_mcp/tools/schemas/channels.py:4-40`
- `src/discord_mcp/tools/schemas/__init__.py:24-44`
- `src/discord_mcp/tools/handlers/router.py:246-256`

**Verify**:
- [ ] Run: `pytest tests/test_entrypoint_wiring.py -q` → pass

---

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

### 4. Update docs for channel CRUD/admin tools

**Depends on**: 3

**Files:**
- Modify: `docs/tool-catalog.md`
- Modify: `README.md`

**What to do**:
- Step 1: Add tool docs for new create/update tools and field contracts.
- Step 2: Document CRUD mapping explicitly (Create/Read/Update/Delete).
- Step 3: Include explicit note for forum field fallback behavior.

**Must NOT do**:
- Must not document fields not supported by implementation.

**References**:
- `docs/tool-catalog.md`
- `README.md:18-20`

**Verify**:
- [ ] Run: `pytest tests/test_channel_admin_tools.py tests/test_entrypoint_wiring.py -q` → pass

---

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

### 6. Final regression and handoff notes

**Depends on**: 5

**Files:**
- Modify: `README.md` (final pointers if needed)
- Modify: `docs/README.md` (final index consistency)

**What to do**:
- Step 1: Run full regression checks.
- Step 2: Summarize final tool list and docs migration map in handoff report.

**Must NOT do**:
- Must not claim completion without passing command evidence.

**Verify**:
- [ ] Run: `pytest -q` → pass
- [ ] Run: `python -m compileall src` → no syntax errors
