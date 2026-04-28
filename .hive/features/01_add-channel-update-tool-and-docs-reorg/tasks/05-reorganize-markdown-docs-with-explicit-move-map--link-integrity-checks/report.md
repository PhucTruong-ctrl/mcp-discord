# Task Report: 05-reorganize-markdown-docs-with-explicit-move-map--link-integrity-checks

**Feature:** add-channel-update-tool-and-docs-reorg
**Completed:** 2026-04-20T10:09:37.246Z
**Status:** success
**Commit:** e272e29f2024420ac7197fd382dcd7161d303297

---

## Summary

Reorganized the markdown docs into product and meta subtrees, added docs/README.md, and updated README links to the moved catalog/rollout/safety pages. Verified local markdown links with the provided script and ran the test suite with uv run --python 3.12 pytest -q (114 passed).

---

## Changes

- **Files changed:** 6
- **Insertions:** +19
- **Deletions:** -3

### Files Modified

- `README.md`
- `docs/README.md`
- `.../plans/2026-03-16-single-guild-lock-design.md`
- `docs/{waves => product/rollout}/01-10-rollout.md`
- `docs/{ => product}/safety/destructive-actions-policy.md`
- `docs/{ => product}/tool-catalog.md`
