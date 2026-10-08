# Discord MCP Server

[![smithery badge](https://smithery.ai/badge/@hanweg/mcp-discord)](https://smithery.ai/server/@hanweg/mcp-discord)
A Model Context Protocol (MCP) server that provides Discord integration capabilities to MCP clients like Claude Desktop.

<a href="https://glama.ai/mcp/servers/wvwjgcnppa"><img width="380" height="200" src="https://glama.ai/mcp/servers/wvwjgcnppa/badge" alt="mcp-discord MCP server" /></a>

- **Target scope**: 212 canonical tools (116 pre-existing + 96 from the discord.py coverage-gap work documented in `docs/product/coverage-gaps-implementation.md`).
- **Current branch registry snapshot**: 212 canonical tools.
- **Rollout model**: 10 implementation waves (Waves 1-10), plus Wave 0 (channel admin), 15 post-wave expansion fillers and 3 post-wave permission/mass-mention tools, with Wave 11 explicitly deferred for stateful extensions

For full details, use:

- [`docs/product/tool-catalog.md`](docs/product/tool-catalog.md) — canonical catalog by domain, baseline vs expansion mapping
- [`docs/analysis/FEATURE_AUDIT.md`](docs/analysis/FEATURE_AUDIT.md)
- [`docs/analysis/DISCORDPY_COVERAGE_GAPS.md`](docs/analysis/DISCORDPY_COVERAGE_GAPS.md) — what discord.py can do that this MCP does not expose yet (list only) — per-tool feature audit (confirmation model + verification evidence)
- [`docs/product/rollout/01-10-rollout.md`](docs/product/rollout/01-10-rollout.md) — wave-by-wave map and Wave 11 deferral rationale
- [`docs/product/safety/destructive-actions-policy.md`](docs/product/safety/destructive-actions-policy.md) — destructive-action guardrails and `confirm_token` policy
- [`docs/README.md`](docs/README.md) — consolidated docs index and navigation

## Environment

| Variable | Required | Purpose |
|---|---|---|
| `DISCORD_TOKEN` | yes | bot token; validated at runtime, never at import |
| `DISCORD_MCP_CONFIRM_SECRET` | yes for guarded tools | HMAC secret for the `dry_run` + `confirm_token` gate (`moderation_*`, `bulk_ban_members`, `prune_inactive_members`, `delete_category`, `incident_*_lockdown`, `automod_apply_ruleset`, `*_roles_bulk`). Without it those tools fail with `DISCORD_MCP_CONFIRM_SECRET environment variable is required ...` and apply nothing |
| `DEFAULT_GUILD_ID` / `DISCORD_GUILD_ID` | no | default server when a tool call omits `server_id` |
| `DISCORD_MCP_STATE_DIR` | no | where incident/tool state is stored (default `~/.local/state/discord-mcp`) |


## Runtime requirements

| Package | Floor | Notes |
|---|---|---|
| Python | 3.12 | `requires-python = ">=3.12"` |
| `discord.py` | 2.7.1 | current PyPI latest is also 2.7.1 |
| `mcp` | **2.3.0** | the 2.x line is required: SDK 2.0 removed the low-level `Server.list_tools()` / `Server.call_tool()` decorators in favour of the `on_list_tools` / `on_call_tool` constructor callbacks that `src/discord_mcp/server.py` uses |

SDK 2.x also renamed pydantic fields to snake_case while keeping the camelCase
wire alias, so `Tool(..., inputSchema=...)` still works but reads must go through
`tool.input_schema`. A failing tool is returned as a result with `is_error` set
rather than as a transport fault, which is what keeps "unknown tool" / "channel
not found" readable to the model.


## Channel CRUD/Admin Mapping

Channel CRUD/admin tools are exposed per channel type:

- Create: `create_text_channel`, `create_voice_channel`, `create_forum_channel`
- Read: `get_channels`, `get_channels_structured`, `get_channel_hierarchy`, `get_channel_type_counts`
- Update: `update_text_channel`, `update_voice_channel`, `update_forum_channel`
- Delete: `delete_channel`

See `docs/product/tool-catalog.md` for the field contracts.
Note: on discord.py 2.7.1+ the `update_forum_channel` handler passes `default_sort_order` through to `ForumChannel.edit(...)`, so the field is supported by the current runtime baseline.

## New Tools Added (84 expansion tools)

The original baseline compatibility surface of 22 tools is now **23 tools** with the addition of `reply_message`:
- #14 (new): `reply_message` — Reply to a specific message using Discord's inline reply threading

The expansion adds these 84 tools:

### Wave 1 — Structured discovery & inventory (8)

1. `get_channels_structured`
2. `get_channel_hierarchy`
3. `get_role_hierarchy`
4. `get_permission_overwrites`
5. `diff_channel_permissions`
6. `export_server_snapshot`
7. `get_channel_type_counts`
8. `list_inactive_channels`

### Wave 2 — Forum/thread intelligence (8)

9. `list_forum_posts`
10. `read_forum_post_messages`
11. `read_forum_posts_batch`
12. `get_thread_context`
13. `list_thread_participants`
14. `get_thread_activity_summary`
15. `tag_forum_post`
16. `retag_forum_post`

### Wave 3 — Moderation core (4)

17. `moderation_bulk_delete`
18. `moderation_timeout_member`
19. `moderation_kick_member`
20. `moderation_ban_member`

### Wave 4 — Channel topology (4)

21. `topology_channel_tree`
22. `topology_channel_children`
23. `topology_role_hierarchy`
24. `topology_permission_matrix`

### Wave 5 — Role governance (8)

25. `create_role`
26. `delete_role`
27. `update_role`
28. `add_roles_bulk`
29. `remove_roles_bulk`
30. `mute_member_role_based`
31. `unmute_member_role_based`
32. `permission_drift_check`

### Wave 6 — Audit analytics (8)

33. `get_audit_log`
34. `get_member_moderation_history`
35. `get_channel_activity_summary`
36. `get_incident_timeline`
37. `get_audit_actor_summary`
38. `check_audit_reason_compliance`
39. `server_health_check`
40. `governance_evidence_packager`

### Wave 7 — Onboarding & lifecycle (8) — mixed capability

41. `get_guild_welcome_screen` — reads live guild data via gateway
42. `update_guild_welcome_screen` — gateway-dependent
43. `get_guild_onboarding` — gateway-dependent (discord.py 2.7.1+ `Guild.onboarding()`)
44. `update_guild_onboarding` — gateway-dependent (discord.py 2.7.1+ `Guild.edit_onboarding()`)
45. `dynamic_role_provision` — gateway-dependent (live role assignment)
46. `verification_gate_orchestrator` — gateway-independent (local logic only)
47. `progressive_access_unlock` — gateway-independent (local logic only)
48. `onboarding_friction_audit` — gateway-independent (local logic only)

### Wave 8 — Messaging, webhooks, integrations (8)

49. `send_embed_message`
50. `send_rich_announcement`
51. `crosspost_announcement`
52. `create_channel_webhook`
53. `list_channel_webhooks`
54. `execute_channel_webhook`
55. `list_guild_integrations`
56. `get_guild_vanity_url`

### Wave 9 — Incident operations (4) — gateway-independent

57. `incident_get_channel_state`
58. `incident_set_channel_state`
59. `incident_apply_lockdown`
60. `incident_rollback_lockdown`

### Wave 10 — AutoMod policy (4) — mixed gateway support

61. `automod_validate_ruleset` — gateway-independent (local shape validation only)
62. `automod_get_ruleset` — queries live Discord API via gateway when available; returns empty rules otherwise
63. `automod_apply_ruleset` — creates rules via gateway when available; dry_run/confirm_token gated
64. `automod_rollback_ruleset` — dry-run (capability check) supported; execute path returns `not_supported`

### Post-wave expansion fillers/utilities (15)

65. `bulk_ban_members` — live Discord API call
66. `prune_inactive_members` — live Discord API call
67. `remove_member_timeout` — live Discord API call
68. `unban_member` — live Discord API call
69. `create_category` — live Discord API call
70. `rename_category` — live Discord API call
71. `move_category` — live Discord API call
72. `delete_category` — live Discord API call
73. `create_incident_room` — live Discord API call
74. `append_incident_event` — live Discord API call
75. `close_incident` — live Discord API call
76. `list_auto_moderation_rules` — live gateway read (`Guild.fetch_automod_rules()`)
77. `create_auto_moderation_rule` — live gateway create; honors `exempt_roles`/`exempt_channels` (ids or names)
78. `update_auto_moderation_rule` — live gateway edit; partial updates, only supplied keys are sent
79. `automod_export_rules` — live gateway read

> **Note**: The remaining expansion filler/utility tools validate input shapes and either perform the real Discord call (when a gateway is configured) or fail loudly; see `docs/analysis/FEATURE_AUDIT.md` and `tests/test_tool_runtime_contracts.py` for the per-tool contract.

Additional baseline behavior notes:

- **`read_messages`** is now embed-aware — each message response includes a structured embed section (title, description, url, image, thumbnail) as both prose text and JSON, in addition to the standard content/reactions output.
- **`reply_message`** — new baseline tool for replying to specific messages using Discord's inline reply threading; pairs with `send_message` for the two core messaging patterns (send vs reply).

The following tool families have specific capability notes:

- **Wave 7 — Onboarding & lifecycle (41–48):** Most tools require a live gateway. Two (`get_guild_onboarding`, `update_guild_onboarding`) use native discord.py 2.7.1+ APIs (`Guild.onboarding()` / `Guild.edit_onboarding()`) via the live gateway. Three (`verification_gate_orchestrator`, `progressive_access_unlock`, `onboarding_friction_audit`) are gateway-independent local logic tools.
- **Wave 9 — Incident operations (57–60):** Gateway-independent. Use `dry_run`/`confirm_token` for lockdown/rollback but no live Discord API calls.
- **Wave 10 — AutoMod policy (61–64):** Mixed — `automod_validate_ruleset` is gateway-independent; `automod_get_ruleset` and `automod_apply_ruleset` use the live Discord API via gateway (discord.py 2.7.1+ `Guild.fetch_automod_rules()` / `Guild.create_automod_rule()`); `automod_rollback_ruleset` execute path returns `not_supported` (no Discord API primitive for rollback).
- **AutoMod field surface** (shared by `automod_apply_ruleset`, `create_auto_moderation_rule` and `update_auto_moderation_rule`):
  - triggers: `keyword` (`keyword_filter` / `regex_patterns` / `allow_list`), `keyword_preset` (`presets` as bitmask int, names, or API ids `1=profanity, 2=sexual_content, 3=slurs`), `spam`, `mention_spam` (`mention_limit` or the API alias `mention_total_limit`, plus `mention_raid_protection` / `mention_raid_protection_enabled`), `member_profile`.
  - actions: `block_message` (+`custom_message`), `send_alert_message` (requires `channel_id`), `timeout` (`duration` or `duration_seconds`, max 2419200), `block_member_interaction`.
  - exemptions: `exempt_roles` / `exempt_role_ids` and `exempt_channels` / `exempt_channel_ids`, given as ids or names; on update only the supplied keys are sent.

### Coverage-gap domains (96 new tools)
- invites & membership · thread management · messages advanced · channel advanced · members & roles advanced · emoji/sticker/soundboard · webhook management · scheduled events & stage · templates & widget · monetization & app commands.
## Installation

1. Set up your Discord bot:
   - Create a new application at [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a bot and copy the token
   - Enable required privileged intents:
     - MESSAGE CONTENT INTENT
     - PRESENCE INTENT
     - SERVER MEMBERS INTENT
   - Invite the bot to your server using OAuth2 URL Generator

2. Clone and install the package (requires **Python ≥ 3.12**):
```bash
# Clone the repository
git clone https://github.com/hanweg/mcp-discord.git
cd mcp-discord

# Create and activate virtual environment
uv venv --python 3.12
.venv\Scripts\activate # On macOS/Linux, use: source .venv/bin/activate

# Install the package
uv pip install -e .
```

3. Configure Claude Desktop (`%APPDATA%\Claude\claude_desktop_config.json` on Windows, `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):
```json
    "discord": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\PATH\\TO\\mcp-discord",
        "run",
        "mcp-discord"
      ],
      "env": {
        "DISCORD_TOKEN": "your_bot_token"
      }
    }
```

### Installing via Smithery

To install Discord Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@hanweg/mcp-discord):

```bash
npx -y @smithery/cli install @hanweg/mcp-discord --client claude
```

## License

MIT License - see LICENSE file for details.
