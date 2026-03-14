# Waves 01-10 Rollout Map

## Rollout principles

- Keep backward compatibility for the legacy 22-tool surface
- Deliver in small, test-gated waves
- Enforce safety controls (`dry_run`, `reason`, `confirm_token`) on destructive operations
- Keep runtime Discord-native only

## Wave map

| Wave | Domain | Planned Count | Implemented in this branch |
|---|---|---:|---:|
| 01 | Structured discovery & inventory | 8 | 8 |
| 02 | Forum/thread intelligence | 8 | 8 |
| 03 | Moderation core | 8 | 4 |
| 04 | Channel topology | 8 | 4 |
| 05 | Roles governance | 8 | 8 |
| 06 | Audit analytics | 8 | 8 |
| 07 | Onboarding & lifecycle | 8 | 8 |
| 08 | Messaging/webhooks/integrations | 8 | 8 |
| 09 | Incident ops/controls | 8 | 4 |
| 10 | AutoMod policy-as-code | 7 | 4 |

**Planned total (waves 1-10):** 79 new tools  
**Implemented in this branch:** 64 new tools  
**Canonical registry snapshot:** 22 baseline + 64 new = 86 tools

## Wave gating approach

Each wave is intended to ship behind verification gates:

1. Tool schema registration updates
2. Router dispatch wiring updates
3. Focused wave test suite(s)
4. Compatibility checks for legacy aliases/contracts

This prevents big-bang risk and enables incremental release-readiness checks.

## Wave 11 deferral rationale (explicit)

Wave 11 is intentionally deferred because it introduces **stateful orchestration surfaces** that are out of current scope for the 101-tool baseline plan. Deferred examples:

- `schedule_message`, `cancel_scheduled_message`
- `create_runbook`, `list_runbooks`, `execute_runbook_step`

Deferral keeps waves 1-10 focused on deterministic Discord API operations and avoids introducing persistence/scheduler concerns before core tool breadth, compatibility, and safety controls are fully stabilized.

---

# Wave 01-10 Rollout and Release Readiness

## Verification Summary (Task 11)

### Required commands

1. `DISCORD_TOKEN=test-token PYTHONPATH=src uv run --python 3.12 python -m unittest discover -s tests -p "test_*.py" -v`
   - Result: **PASS**
   - Evidence: 90 tests run, 0 failures, 0 errors (`OK`).

2. `PYTHONPATH=src uv run --python 3.12 python -m compileall src`
   - Result: **PASS**
   - Evidence: compileall completed for `src/discord_mcp` tree with no compilation errors.

3. `DISCORD_TOKEN=test-token PYTHONPATH=src uv run --python 3.12 python -c "import discord_mcp.server; print('ok')"`
   - Result: **PASS**
   - Evidence: output is `ok`.

## Pass/Fail by wave

- **Wave 1 (Inventory): PASS**
  - Tool registration and handler tests pass (`test_inventory_tools.py`).
- **Wave 2 (Forum intelligence): PASS**
  - Schema/router wiring and behavior tests pass (`test_forum_intel_tools.py`).
- **Wave 3 (Moderation core): PASS**
  - Confirm-token dry-run/execute gating tests pass (`test_moderation_core_tools.py`).
- **Wave 4 (Topology): PASS**
  - Topology schemas/routes/behaviors pass (`test_topology_tools.py`).
- **Wave 5 (Role governance): PASS**
  - Governance tool behavior and wiring pass (`test_role_governance_tools.py`).
- **Wave 6 (Audit analytics): PASS**
  - Analytics payload and routing tests pass (`test_audit_analytics_tools.py`).
- **Wave 7 (Onboarding): PASS**
  - Stateless onboarding contract and orchestrator tests pass (`test_onboarding_tools.py`).
- **Wave 8 (Messaging workflow): PASS**
  - Messaging/webhook/integration tool tests pass (`test_messaging_workflow_tools.py`).
- **Wave 9 (Incident ops): PASS**
  - Incident state and lockdown confirm-token tests pass (`test_incident_ops_tools.py`).
- **Wave 10 (Automod policy): PASS**
  - Ruleset validation/apply/rollback tests pass (`test_automod_policy_tools.py`).

## Go/No-Go checklist

- [x] Full automated unittest suite passes.
- [x] Source compiles cleanly with `compileall`.
- [x] Server import smoke check passes.
- [x] 22 legacy tools preserve baseline prefix ordering and alias compatibility tests.
- [x] Confirm-token safety behavior is covered and passing in safety-focused suites.
- [ ] Canonical registry reaches **101 tools** target.

## Unresolved risks

1. **Canonical registry count mismatch (blocking):**
   - Observed current canonical count: **86** tools.
   - Planned/required count: **101** tools (22 existing + 79 new).
   - Impact: release does not satisfy full roadmap contract; this is a hard release gate failure.

2. **Readiness decision:**
   - **NO-GO** for final release approval until canonical count reaches 101 and corresponding registry coverage is restored/validated.

## Current state snapshot

- Canonical tools: `86`
- Router keys (including aliases): `103`
- Baseline first 22 canonical names: preserved and verified by tests.
