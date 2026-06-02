# Discord MCP Tool Catalog

## Scope snapshot

- Planned total: **107 canonical tools** (23 baseline + 84 expansion)
- Current canonical registry in this branch: **107 tools**
- Runtime is Discord-native only (Discord API + bot token), no external runtime dependency

## Channel CRUD/admin tools

The channel admin surface is split by channel type and operation:

- Create: `create_text_channel`, `create_voice_channel`, `create_forum_channel`
- Update: `update_text_channel`, `update_voice_channel`, `update_forum_channel`
- Delete: `delete_channel`

Field contracts:

- `create_text_channel`: `server_id`, `name`, optional `category_id`, optional `topic`
- `update_text_channel`: `server_id`, `channel_id`, optional `name`, optional `category_id`, optional `topic`, optional `nsfw`, optional `slowmode_delay`, optional `position`, optional `reason`
- `create_voice_channel`: `server_id`, `name`, optional `category_id`, optional `bitrate`, optional `user_limit`, optional `rtc_region`, optional `video_quality_mode`
- `update_voice_channel`: `server_id`, `channel_id`, optional `name`, optional `category_id`, optional `bitrate`, optional `user_limit`, optional `rtc_region`, optional `video_quality_mode`, optional `position`, optional `reason`
- `create_forum_channel`: `server_id`, `name`, optional `category_id`, optional `topic`, optional `nsfw`, optional `slowmode_delay`, optional `default_auto_archive_duration`, optional `default_reaction_emoji`, optional `default_sort_order`, optional `available_tags`
- `update_forum_channel`: `server_id`, `channel_id`, optional `name`, optional `category_id`, optional `topic`, optional `nsfw`, optional `slowmode_delay`, optional `default_auto_archive_duration`, optional `default_reaction_emoji`, optional `default_sort_order`, optional `available_tags`, optional `position`, optional `reason`

Notes:
- `update_forum_channel` supports `default_sort_order` on the current discord.py 2.7.1+ runtime and passes it through to `ForumChannel.edit(...)`.
- Update tools reject unknown fields with `unsupported_fields: ...`.

## Baseline 23 tools (legacy compatibility surface)

1. `get_server_info`
2. `get_channels`
3. `list_members`
4. `add_role`
5. `remove_role`
6. `create_text_channel`
7. `delete_channel`
8. `add_reaction`
9. `add_multiple_reactions`
10. `remove_reaction`
11. `send_message`
12. `read_messages` — embed-aware (returns structured embed data alongside content/reactions)
13. `edit_message`
14. `reply_message` — reply to a specific message using Discord's inline reply threading
15. `read_forum_threads`
16. `list_threads`
17. `search_threads`
18. `add_thread_tags`
19. `unarchive_thread`
20. `download_attachment`
21. `get_user_info`
22. `moderate_message`
23. `list_servers`

## Expansion catalog (implemented in this branch)

### Wave 0 — Channel admin (5)

23. `create_voice_channel`
24. `create_forum_channel`
25. `update_text_channel`
26. `update_voice_channel`
27. `update_forum_channel`

### Wave 1 — Structured discovery & inventory (8)

28. `get_channels_structured`
29. `get_channel_hierarchy`
30. `get_role_hierarchy`
31. `get_permission_overwrites`
32. `diff_channel_permissions`
33. `export_server_snapshot`
34. `get_channel_type_counts`
35. `list_inactive_channels`

### Wave 2 — Forum/thread intelligence (8)

36. `list_forum_posts`
37. `read_forum_post_messages`
38. `read_forum_posts_batch`
39. `get_thread_context`
40. `list_thread_participants`
41. `get_thread_activity_summary`
42. `tag_forum_post`
43. `retag_forum_post`

### Wave 3 — Moderation core (4 implemented)

44. `moderation_bulk_delete`
45. `moderation_timeout_member`
46. `moderation_kick_member`
47. `moderation_ban_member`

### Wave 4 — Channel topology (4 implemented)

48. `topology_channel_tree`
49. `topology_channel_children`
50. `topology_role_hierarchy`
51. `topology_permission_matrix`

### Wave 5 — Role governance (8)

52. `create_role`
53. `delete_role`
54. `update_role`
55. `add_roles_bulk`
56. `remove_roles_bulk`
57. `mute_member_role_based`
58. `unmute_member_role_based`
59. `permission_drift_check`

### Wave 6 — Audit analytics (8)

60. `get_audit_log`
61. `get_member_moderation_history`
62. `get_channel_activity_summary`
63. `get_incident_timeline`
64. `get_audit_actor_summary`
65. `check_audit_reason_compliance`
66. `server_health_check`
67. `governance_evidence_packager`

### Wave 7 — Onboarding & lifecycle (8)

69. `get_guild_welcome_screen`
70. `update_guild_welcome_screen`
71. `get_guild_onboarding`
72. `update_guild_onboarding`
73. `dynamic_role_provision`
74. `verification_gate_orchestrator`
75. `progressive_access_unlock`
76. `onboarding_friction_audit`

### Wave 8 — Messaging, webhooks, integrations (8)

77. `send_embed_message`
78. `send_rich_announcement`
79. `crosspost_announcement`
80. `create_channel_webhook`
81. `list_channel_webhooks`
82. `execute_channel_webhook`
83. `list_guild_integrations`
84. `get_guild_vanity_url`

### Wave 9 — Incident operations (4 implemented)

85. `incident_get_channel_state`
86. `incident_set_channel_state`
87. `incident_apply_lockdown`
88. `incident_rollback_lockdown`

### Wave 10 — AutoMod policy (4 implemented)

89. `automod_validate_ruleset`
90. `automod_get_ruleset`
91. `automod_apply_ruleset`
92. `automod_rollback_ruleset`

### Post-wave expansion fillers/utilities (15)

93. `bulk_ban_members`
94. `prune_inactive_members`
95. `remove_member_timeout`
96. `unban_member`
97. `create_category`
98. `rename_category`
99. `move_category`
100. `delete_category`
101. `create_incident_room`
102. `append_incident_event`
103. `close_incident`
104. `list_auto_moderation_rules`
105. `create_auto_moderation_rule`
106. `update_auto_moderation_rule`
107. `automod_export_rules`

## Implementation-status note

The 15 expansion filler/utility tools split into two groups with different runtime behavior:

- **Tools 93–103 (synthetic-only):** `bulk_ban_members`, `prune_inactive_members`, `remove_member_timeout`, `unban_member`, `create_category`, `rename_category`, `move_category`, `delete_category`, `create_incident_room`, `append_incident_event`, `close_incident` — return synthetic/placeholder responses. They validate input shapes and may use the `dry_run`/`confirm_token` safety pattern for destructive operations, but do **not** make live Discord API calls.
- **Tools 104–107 (gateway-aware with synthetic fallback):** `list_auto_moderation_rules`, `create_auto_moderation_rule`, `update_auto_moderation_rule`, `automod_export_rules` — use the live Discord API via gateway when available; return synthetic placeholder responses when gateway is absent.

This preserves the full 107-tool registry contract while deeper Discord side-effect implementations for the synthetic-only tools continue in follow-up work.

The following tool families have specific capability notes:

- **Wave 7 — Onboarding & lifecycle (69–76):** Most tools require a live gateway. Two (`get_guild_onboarding`, `update_guild_onboarding`) now use native discord.py 2.7.1+ Guild.onboarding() and Guild.edit_onboarding() APIs. Three (`verification_gate_orchestrator`, `progressive_access_unlock`, `onboarding_friction_audit`) are gateway-independent local logic tools.
- **Wave 9 — Incident operations (85–88):** Gateway-independent. Use `dry_run`/`confirm_token` for lockdown/rollback but no live Discord API calls.
- **Wave 10 — AutoMod policy (89–92):** Mixed — `automod_validate_ruleset` is gateway-independent; `automod_get_ruleset` and `automod_apply_ruleset` use the live Discord API via gateway when available; `automod_rollback_ruleset` execute path returns `not_supported` (no Discord API primitive for rollback).
- **Expansion fillers — tools 93–103 (synthetic-only):** Return `"applied"` or `"ok"` responses without Discord API calls.
- **Expansion fillers — tools 104–107 (gateway-aware):** Use Discord API via gateway when available; fall back to synthetic `"applied"`/`"ok"` responses when gateway is absent.

## 107-tool contract status

The canonical registry target is restored in this branch: **107 canonical tools** (23 baseline + 84 expansion). All tools are covered by registry-count, router-coverage, and runtime-contract tests. See `tests/test_tool_runtime_contracts.py` for the detailed contract assertions.
