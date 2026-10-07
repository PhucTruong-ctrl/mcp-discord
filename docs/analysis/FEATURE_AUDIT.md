# Feature audit - all 110 MCP tools

Audit date: 2026-10-07. Target server: `1424116735782682778` (live Discord API, discord.py 2.7.1).

Columns: **gate** = confirmation model, **verified** = evidence level.

| # | tool | family | gate | verified | note |
|---|---|---|---|---|---|
| 1 | `get_server_info` | Baseline: server/member reads | direct | live |  |
| 2 | `get_channels` | Channels | direct | live |  |
| 3 | `list_members` | Baseline: server/member reads | direct | live |  |
| 4 | `add_role` | Baseline: roles | direct | unit test |  |
| 5 | `remove_role` | Baseline: roles | direct | unit test |  |
| 6 | `create_text_channel` | Channels | direct | unit test |  |
| 7 | `delete_channel` | Channels | direct | unit test |  |
| 8 | `add_reaction` | Misc | direct | live |  |
| 9 | `add_multiple_reactions` | Misc | direct | unit test |  |
| 10 | `remove_reaction` | Misc | direct | unit test |  |
| 11 | `send_message` | Messages | direct | live |  |
| 12 | `read_messages` | Messages | direct | live |  |
| 13 | `edit_message` | Messages | direct | live |  |
| 14 | `reply_message` | Messages | direct | live |  |
| 15 | `read_forum_threads` | Forums/threads | direct | live |  |
| 16 | `list_threads` | Forums/threads | direct | live |  |
| 17 | `search_threads` | Forums/threads | direct | live |  |
| 18 | `add_thread_tags` | Forums/threads | direct | unit test |  |
| 19 | `unarchive_thread` | Forums/threads | direct | unit test |  |
| 20 | `download_attachment` | Misc | local only | live |  |
| 21 | `get_user_info` | Misc | direct | live |  |
| 22 | `moderate_message` | Misc | direct | live |  |
| 23 | `list_servers` | Baseline: server/member reads | direct | live |  |
| 24 | `create_voice_channel` | Channels | direct | unit test |  |
| 25 | `create_forum_channel` | Channels | direct | unit test |  |
| 26 | `update_text_channel` | Channels | direct | unit test |  |
| 27 | `update_voice_channel` | Channels | direct | unit test |  |
| 28 | `update_forum_channel` | Channels | direct | unit test |  |
| 29 | `list_forum_posts` | Forum intel | direct | live |  |
| 30 | `read_forum_post_messages` | Forum intel | direct | unit test |  |
| 31 | `read_forum_posts_batch` | Forum intel | direct | unit test |  |
| 32 | `get_thread_context` | Forum intel | direct | unit test |  |
| 33 | `list_thread_participants` | Forum intel | direct | unit test |  |
| 34 | `get_thread_activity_summary` | Forum intel | direct | live |  |
| 35 | `tag_forum_post` | Forum intel | direct | unit test |  |
| 36 | `retag_forum_post` | Forum intel | direct | unit test |  |
| 37 | `get_channels_structured` | Inventory/permissions | direct | live |  |
| 38 | `get_channel_hierarchy` | Inventory/permissions | direct | live |  |
| 39 | `get_role_hierarchy` | Inventory/permissions | direct | live |  |
| 40 | `get_permission_overwrites` | Inventory/permissions | direct | live |  |
| 41 | `diff_channel_permissions` | Inventory/permissions | direct | live |  |
| 42 | `export_server_snapshot` | Inventory/permissions | direct | live |  |
| 43 | `get_channel_type_counts` | Inventory/permissions | direct | live |  |
| 44 | `list_inactive_channels` | Inventory/permissions | direct | live |  |
| 45 | `moderation_bulk_delete` | Moderation core | dry_run + confirm_token | live |  |
| 46 | `moderation_timeout_member` | Moderation core | dry_run + confirm_token | unit test | execute needs a real member; gateway call unit-tested (member.timeout) |
| 47 | `moderation_kick_member` | Moderation core | dry_run + confirm_token | unit test | execute needs a real member; gateway call unit-tested (member.kick) |
| 48 | `moderation_ban_member` | Moderation core | dry_run + confirm_token | unit test | execute needs a real member; gateway call unit-tested (guild.ban) |
| 49 | `topology_channel_tree` | Topology | direct | live |  |
| 50 | `topology_channel_children` | Topology | direct | live |  |
| 51 | `topology_role_hierarchy` | Topology | direct | live |  |
| 52 | `topology_permission_matrix` | Topology | direct | live |  |
| 53 | `create_role` | Role governance | direct | live |  |
| 54 | `delete_role` | Role governance | direct | live |  |
| 55 | `update_role` | Role governance | direct | live |  |
| 56 | `add_roles_bulk` | Role governance | dry_run + confirm_token | live |  |
| 57 | `remove_roles_bulk` | Role governance | dry_run + confirm_token | live |  |
| 58 | `mute_member_role_based` | Role governance | direct | live |  |
| 59 | `unmute_member_role_based` | Role governance | direct | live |  |
| 60 | `permission_drift_check` | Role governance | direct | live |  |
| 61 | `get_audit_log` | Audit analytics | direct | live |  |
| 62 | `get_member_moderation_history` | Audit analytics | direct | live |  |
| 63 | `get_channel_activity_summary` | Audit analytics | direct | live |  |
| 64 | `get_incident_timeline` | Audit analytics | direct | live |  |
| 65 | `get_audit_actor_summary` | Audit analytics | direct | live |  |
| 66 | `check_audit_reason_compliance` | Audit analytics | direct | live |  |
| 67 | `server_health_check` | Audit analytics | direct | live |  |
| 68 | `governance_evidence_packager` | Audit analytics | direct | live |  |
| 69 | `get_guild_welcome_screen` | Onboarding | direct | unit test |  |
| 70 | `update_guild_welcome_screen` | Onboarding | direct | unit test |  |
| 71 | `get_guild_onboarding` | Onboarding | direct | live |  |
| 72 | `update_guild_onboarding` | Onboarding | direct | unit test |  |
| 73 | `dynamic_role_provision` | Onboarding | direct | unit test |  |
| 74 | `verification_gate_orchestrator` | Onboarding | local only | unit test |  |
| 75 | `progressive_access_unlock` | Onboarding | local only | unit test |  |
| 76 | `onboarding_friction_audit` | Onboarding | local only | unit test |  |
| 77 | `send_embed_message` | Messaging/workflow | direct | unit test |  |
| 78 | `send_rich_announcement` | Messaging/workflow | direct | unit test |  |
| 79 | `crosspost_announcement` | Messaging/workflow | direct | unit test |  |
| 80 | `create_channel_webhook` | Messaging/workflow | direct | unit test |  |
| 81 | `list_channel_webhooks` | Messaging/workflow | direct | live |  |
| 82 | `execute_channel_webhook` | Messaging/workflow | direct | unit test |  |
| 83 | `list_guild_integrations` | Messaging/workflow | direct | live |  |
| 84 | `get_guild_vanity_url` | Messaging/workflow | direct | live |  |
| 85 | `incident_get_channel_state` | Incident ops | local only | live |  |
| 86 | `incident_set_channel_state` | Incident ops | local only | live |  |
| 87 | `incident_apply_lockdown` | Incident ops | dry_run + confirm_token | live |  |
| 88 | `incident_rollback_lockdown` | Incident ops | dry_run + confirm_token | live |  |
| 89 | `automod_validate_ruleset` | AutoMod policy | local only | unit test |  |
| 90 | `automod_get_ruleset` | AutoMod policy | direct | live |  |
| 91 | `automod_apply_ruleset` | AutoMod policy | dry_run + confirm_token | live |  |
| 92 | `automod_rollback_ruleset` | AutoMod policy | dry_run + confirm_token | unit test | Discord has no rollback API: returns not_supported by design |
| 93 | `remove_member_timeout` | Expansion utilities | direct | live(api reached) |  |
| 94 | `unban_member` | Expansion utilities | direct | live(error path) |  |
| 95 | `bulk_ban_members` | Expansion utilities | dry_run + confirm_token | unit test | would ban real users; guild.bulk_ban unit-tested + dry-run verified live |
| 96 | `prune_inactive_members` | Expansion utilities | dry_run + confirm_token | unit test | would prune real users; guild.prune_members unit-tested + dry-run verified live |
| 97 | `create_category` | Expansion utilities | direct | live |  |
| 98 | `rename_category` | Expansion utilities | direct | live |  |
| 99 | `move_category` | Expansion utilities | direct | live |  |
| 100 | `delete_category` | Expansion utilities | dry_run + confirm_token | live |  |
| 101 | `create_incident_room` | Expansion utilities | direct | live |  |
| 102 | `append_incident_event` | Expansion utilities | direct | live |  |
| 103 | `close_incident` | Expansion utilities | direct | live |  |
| 104 | `list_auto_moderation_rules` | Expansion utilities | direct | live |  |
| 105 | `create_auto_moderation_rule` | Expansion utilities | direct | unit test |  |
| 106 | `update_auto_moderation_rule` | Expansion utilities | direct | unit test |  |
| 107 | `automod_export_rules` | Expansion utilities | direct | live |  |
| 108 | `get_role_permissions` | Permission intel | direct | live |  |
| 109 | `compute_member_permissions` | Permission intel | direct | live |  |
| 110 | `audit_mass_mentions` | Mass mentions | direct | live |  |

## Findings fixed in this audit

- 11 expansion utilities reported `{"status": "applied"}` without calling Discord at all;
  3 of them lied again after a valid `confirm_token`. All 11 now perform the real API call
  (or fail loudly when no gateway is configured).
- 4 moderation tools called gateway methods that did not exist (`bulk_delete_messages`,
  `timeout_member`, `kick_member`, `ban_member`) - the execute path would raise AttributeError.
  The gateway now implements all four.
- Incident lockdown/rollback only echoed a payload after confirmation, and `incident_get/set_channel_state`
  stored nothing. They now snapshot the real `@everyone` overwrite, change it, persist state in
  `$DISCORD_MCP_STATE_DIR/state.json` and restore the snapshot on rollback (verified live).
- AutoMod tools returned synthetic `applied`/empty results without a gateway; they now raise
  `ValueError: gateway is required ...` instead of pretending.
- `get_guild_welcome_screen` / `get_guild_onboarding` surfaced a raw 404 for servers without the
  feature; they now explain what is missing.
- `automod_apply_ruleset` did not wire `exempt_roles`/`exempt_channels`; keyword triggers now also
  accept `allow_list` / `regex_patterns`.

## Not run against real members (by design)

Ban / kick / timeout / prune execute paths need a real victim; the gateway calls are unit-tested and
the dry-run + confirm_token gates were exercised live. `automod_rollback_ruleset` returns
`not_supported` because Discord exposes no rollback primitive.
