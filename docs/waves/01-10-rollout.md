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
