## At a Glance
- Plan revised after Hygienic review: now includes explicit supported-field contract, forum unsupported-field fallback behavior, explicit docs move map, and deterministic markdown link integrity verification.
- Deliverables remain: add channel admin create/update tools for text/voice/forum + move docs markdown under `docs/` hierarchy.

## Workstreams
1) Tooling & Tests
- Add/green tests for registry/router + per-type update validations.
- Implement handlers with strict unsupported-field errors.

2) Documentation
- Update tool docs + CRUD coverage statement.
- Move markdown files via explicit source->destination map.
- Fix local links with script-based validation.

3) Final Verification
- Full `pytest -q` and compile checks.

## Revision History
- 2026-04-20: Initial plan drafted.
- 2026-04-20: Hygienic review requested changes (forum field support + docs map + read coverage). Plan updated accordingly.