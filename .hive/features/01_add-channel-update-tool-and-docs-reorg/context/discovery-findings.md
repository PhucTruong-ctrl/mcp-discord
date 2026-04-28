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