# Discord MCP Tool Catalog

## Scope snapshot

- Planned total: **101 canonical tools** (22 baseline + 79 expansion)
- Current canonical registry in this branch: **101 tools**
- Runtime is Discord-native only (Discord API + bot token), no external runtime dependency

## Channel CRUD/admin tools

The channel admin surface is split by channel type and operation:

- Create: `create_text_channel`, `create_voice_channel`, `create_forum_channel`
- Update: `update_text_channel`, `update_voice_channel`, `update_forum_channel`
- Delete: `delete_channel`

Field contracts:

- `create_text_channel`: `server_id`, `name`, optional `category_id`, optional `topic`
- `update_text_channel`: `channel_id`, optional `name`, optional `category_id`, optional `topic`
- `create_voice_channel`: `server_id`, `name`, optional `category_id`, optional `bitrate`, optional `user_limit`
- `update_voice_channel`: `channel_id`, optional `name`, optional `category_id`, optional `bitrate`, optional `user_limit`
- `create_forum_channel`: `server_id`, `name`, optional `category_id`, optional forum text field
- `update_forum_channel`: `channel_id`, optional `name`, optional `category_id`, optional forum text field

Forum channel text fields follow the handler fallback behavior: callers may provide the forum text value using the supported alias, and the implementation normalizes it before sending the update.

## Baseline 22 tools (legacy compatibility surface)

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
12. `read_messages`
13. `edit_message`
14. `read_forum_threads`
15. `list_threads`
16. `search_threads`
17. `add_thread_tags`
18. `unarchive_thread`
19. `download_attachment`
20. `get_user_info`
21. `moderate_message`
22. `list_servers`

## Expansion catalog (implemented in this branch)

### Wave 1 — Structured discovery & inventory (8)

23. `get_channels_structured`
24. `get_channel_hierarchy`
25. `get_role_hierarchy`
26. `get_permission_overwrites`
27. `diff_channel_permissions`
28. `export_server_snapshot`
29. `get_channel_type_counts`
30. `list_inactive_channels`

### Wave 2 — Forum/thread intelligence (8)

31. `list_forum_posts`
32. `read_forum_post_messages`
33. `read_forum_posts_batch`
34. `get_thread_context`
35. `list_thread_participants`
36. `get_thread_activity_summary`
37. `tag_forum_post`
38. `retag_forum_post`

### Wave 3 — Moderation core (4 implemented)

39. `moderation_bulk_delete`
40. `moderation_timeout_member`
41. `moderation_kick_member`
42. `moderation_ban_member`

### Wave 4 — Channel topology (4 implemented)

43. `topology_channel_tree`
44. `topology_channel_children`
45. `topology_role_hierarchy`
46. `topology_permission_matrix`

### Wave 5 — Role governance (8)

47. `create_role`
48. `delete_role`
49. `update_role`
50. `add_roles_bulk`
51. `remove_roles_bulk`
52. `mute_member_role_based`
53. `unmute_member_role_based`
54. `permission_drift_check`

### Wave 6 — Audit analytics (8)

55. `get_audit_log`
56. `get_member_moderation_history`
57. `get_channel_activity_summary`
58. `get_incident_timeline`
59. `get_audit_actor_summary`
60. `check_audit_reason_compliance`
61. `server_health_check`
62. `governance_evidence_packager`

### Wave 7 — Onboarding & lifecycle (8)

63. `get_guild_welcome_screen`
64. `update_guild_welcome_screen`
65. `get_guild_onboarding`
66. `update_guild_onboarding`
67. `dynamic_role_provision`
68. `verification_gate_orchestrator`
69. `progressive_access_unlock`
70. `onboarding_friction_audit`

### Wave 8 — Messaging, webhooks, integrations (8)

71. `send_embed_message`
72. `send_rich_announcement`
73. `crosspost_announcement`
74. `create_channel_webhook`
75. `list_channel_webhooks`
76. `execute_channel_webhook`
77. `list_guild_integrations`
78. `get_guild_vanity_url`

### Wave 9 — Incident operations (4 implemented)

79. `incident_get_channel_state`
80. `incident_set_channel_state`
81. `incident_apply_lockdown`
82. `incident_rollback_lockdown`

### Wave 10 — AutoMod policy (4 implemented)

83. `automod_validate_ruleset`
84. `automod_get_ruleset`
85. `automod_apply_ruleset`
86. `automod_rollback_ruleset`

### Post-wave expansion fillers/utilities (15)

87. `bulk_ban_members`
88. `prune_inactive_members`
89. `remove_member_timeout`
90. `unban_member`
91. `create_category`
92. `rename_category`
93. `move_category`
94. `delete_category`
95. `create_incident_room`
96. `append_incident_event`
97. `close_incident`
98. `list_auto_moderation_rules`
99. `create_auto_moderation_rule`
100. `update_auto_moderation_rule`
101. `automod_export_rules`

## Implementation-status note

Tools 87-101 are currently implemented as expansion filler/utility handlers; a subset are lightweight stubs intended to preserve the 101-tool registry contract while deeper Discord side-effect implementations continue in follow-up work.

## 101-tool contract status

The canonical registry target is restored in this branch: **101 canonical tools** (22 baseline + 79 expansion). The filler/utility tools (87-101) are now included in the catalog and are expected to remain covered by registry-count and router-coverage tests.
