# Discord MCP Tool Catalog

## Scope snapshot

- Planned total: **115 canonical tools** (24 baseline + 86 expansion + 2 permission introspection + 1 mass-mention audit + 2 member admin)
- Current canonical registry in this branch: **115 tools**
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

### Post-wave additions — permission introspection (2)

108. `get_role_permissions`
109. `compute_member_permissions`

### Post-wave addition — mass-mention audit (1)

110. `audit_mass_mentions`

### Post-wave addendum — guild settings & channel overwrites (3)

111. `update_guild` — `description` (null clears), `verification_level`
     (none/low/medium/high/highest or 0-4), `explicit_content_filter`
     (disabled/no_role/all_members or 0-2), optional `reason`
112. `set_channel_permission_overwrite` — `channel_id`, `target_id`, optional `target_type`
     (role|member, auto-detected from cache when omitted), optional `allow`/`deny` arrays of
     permission names or raw bit values, optional `reason`. Replaces the overwrite for that target
113. `remove_channel_permission_overwrite` — `channel_id`, `target_id`, optional `target_type`,
     optional `reason`. Deletes the explicit overwrite; inherited state is untouched

### Post-wave addition — member admin (2)

114. `set_member_roles`
115. `set_member_nickname`

- **Role sets and role references:** `set_member_roles(member_id, role_ids)` replaces a member's whole role set
  in one `dry_run` + `confirm_token` call (empty array clears roles; @everyone is ignored; integration-managed
  roles are rejected when assigning and preserved when replacing, because Discord refuses a request that drops
  them). `update_role` / `delete_role` accept `role_name` as an alternative to `role_id` (unique, case-insensitive),
  and a null `color` / `secondary_color` / `tertiary_color` clears that colour.
- **Role colours and role assignment:** every role-emitting tool (`get_role_permissions`,
  `get_role_hierarchy`, `topology_role_hierarchy`, `topology_permission_matrix`,
  `export_server_snapshot`, `permission_drift_check`) now reports `color` (int), `colorHex`,
  `secondaryColor`/`tertiaryColor` and `gradient`. `create_role` / `update_role` accept `color`
  plus `secondary_color`/`tertiary_color` as int, `#rrggbb` or `0xrrggbb`; gradient colours need a
  Discord guild feature, and the tool explains the `670006 Missing guild feature` refusal.
  `add_role` / `remove_role` resolve uncached roles, take an optional `reason`, report
  `hadRoleBefore` / `hasRoleNow` / `changed` from a fresh API read (discord.py's `Member.edit`
  returns a new object and leaves the instance you called it on stale) and name MANAGE_ROLES plus
  the role-hierarchy comparison when Discord answers 403.
## What `update_guild` can and cannot change

`update_guild` maps the whole `discord.py Guild.edit` surface (PATCH /guilds/{id}):

- text: `name`, `description`, `preferred_locale`, `vanity_code`
- enums: `verification_level`, `explicit_content_filter`, `default_notifications`, `mfa_level`
- booleans: `community`, `discoverable`, `invites_disabled`, `widget_enabled`,
  `premium_progress_bar_enabled`, `raid_alerts_disabled`
- channels (id or name): `afk_channel`, `system_channel`, `rules_channel`,
  `public_updates_channel`, `safety_alerts_channel`, `widget_channel`; plus `afk_timeout`
  (60/300/900/1800/3600), `system_channel_flags` (bitfield or flag names) and `owner`
- images (http(s) URL, data URI or local path, `null` clears): `icon`, `banner`, `splash`,
  `discovery_splash` — each needs its guild feature (`ANIMATED_ICON`, `BANNER`/`ANIMATED_BANNER`,
  `INVITE_SPLASH`, `DISCOVERABLE`)
- timestamps: `invites_disabled_until`, `dms_disabled_until`

Unknown fields are rejected (`unsupported_fields: ...`). **Not settable through the API** (client-only
Server Profile features, no documented fields or endpoints): the profile **banner colour**, **traits**,
**games**, **private profile**, and the **server tag** (the tag lives on the *user* as `primary_guild`).
The API `banner` field is an image, not the profile colour banner.

## Implementation-status note

All 15 expansion utilities (tools 93–107) make live Discord API calls — see the
*Expansion utilities — tools 93–107* note below. There are no synthetic/placeholder handlers left in
the registry; every tool either performs the documented Discord operation or fails with an explicit
error.

The following tool families have specific capability notes:

- **Onboarding writes (`update_guild_onboarding`):** the payload is converted into
  `discord.OnboardingPrompt`/`OnboardingPromptOption` objects (raw JSON used to raise
  `'dict' object has no attribute 'to_dict'`) and merged onto the current configuration, because the
  endpoint is a PUT that empties any omitted field. Discord itself rejects the write with
  `350001 Cannot update onboarding while below requirements` unless the server has at least 7 public
  channels and at least 5 of them writable by `@everyone`; the tool reports the guild's own counts in
  that error, and the Discord client is blocked by the same rule.
- **Wave 7 — Onboarding & lifecycle (69–76):** Most tools require a live gateway. Two (`get_guild_onboarding`, `update_guild_onboarding`) now use native discord.py 2.7.1+ Guild.onboarding() and Guild.edit_onboarding() APIs. Three (`verification_gate_orchestrator`, `progressive_access_unlock`, `onboarding_friction_audit`) are gateway-independent local logic tools.
- **Wave 9 — Incident operations (85–88):** `dry_run`/`confirm_token` gated; the confirmed paths change real channel overwrites and persist state (see *Incident state* below).
- **AutoMod exemptions (`automod_apply_ruleset`):** each rule accepts `exempt_roles` and
  `exempt_channels` as ids or names (max 20 roles / 50 channels), and keyword triggers accept
  `keyword_filter`, `regex_patterns` and `allow_list`. Practical use: block `*@everyone*` /
  `*@here*` text for everyone while exempting the roles that legitimately mass-mention.
  Platform caveat verified on live Discord: AutoMod does not act on bot messages, so such a
  rule constrains members, not bots.
- **Wave 10 — AutoMod policy (89–92):** Mixed — `automod_validate_ruleset` is gateway-independent; `automod_get_ruleset` and `automod_apply_ruleset` use the live Discord API via gateway when available; `automod_rollback_ruleset` execute path returns `not_supported` (no Discord API primitive for rollback).
- **Expansion utilities — tools 93–107 (live Discord API):** every one performs the real call:
  member/category/incident-room CRUD goes through discord.py, bulk ban uses `guild.bulk_ban`, pruning uses
  `guild.prune_members`, and the AutoMod tools read/write rules. Destructive ones keep the
  `dry_run` + `confirm_token` gate (`bulk_ban_members`, `prune_inactive_members`, `delete_category`).
  With no gateway configured they raise `ValueError: gateway is required ...` instead of reporting a
  fake success.
- **Incident state:** `incident_apply_lockdown` snapshots the channel's `@everyone` overwrite, denies
  `send_messages` / `send_messages_in_threads` / `create_public_threads`, and records the snapshot in
  `$DISCORD_MCP_STATE_DIR/state.json` (default `~/.local/state/discord-mcp`); `incident_rollback_lockdown`
  restores it. `incident_get/set_channel_state` read/write the same store.
- **Confirm-token secret:** every `dry_run`/`confirm_token` tool requires `DISCORD_MCP_CONFIRM_SECRET`
  in the environment; without it those tools fail with a clear error and nothing is applied.

## Permission introspection (tools 108-109)

Role and channel permission reads all share one payload shape (`src/discord_mcp/core/permissions.py`):

- Role rows: `id`, `name`, `position`, `permissions` (int bitfield), `permissionNames` (decoded
  flags), `hoist`, `mentionable`, `managed`. Emitted by `get_role_permissions`,
  `get_role_hierarchy`, `topology_role_hierarchy`, `topology_permission_matrix`,
  `export_server_snapshot` and `permission_drift_check`.
- Overwrite rows: `targetId`, `targetName`, `targetType`, `allow`, `deny` (int masks) plus
  `allowNames`/`denyNames`. Emitted by `get_permission_overwrites`, `diff_channel_permissions`
  and `topology_permission_matrix`.
- `get_role_permissions`: every role (optionally one `role_id`) with decoded bitfields.
- `compute_member_permissions`: base and effective bitfields plus the layer that decided each
  permission (`administrator`, `base_role`, `everyone_overwrite`, `role_overwrite:<roleId>`,
  `member_overwrite`, each suffixed `:allow`/`:deny`). The walk matches discord.py's
  `GuildChannel.permissions_for` overwrite resolution; it does not apply discord.py's
  implicit per-channel-type flag stripping, and category overwrites are reported
  (`categoryOverwrites`) rather than inherited.
- Drift round-trip: `export_server_snapshot` (v2) emits role permission bitfields, so its
  payload is a valid `permission_drift_check` baseline; passing the same payload straight back
  yields `driftCount: 0`.

## Mass mentions (tool 110)

`audit_mass_mentions` scans channel history (optionally threads/forum posts) and separates
two things that look identical in a chat client:

- `kind: "delivered"` — the message has `mention_everyone: true`; Discord registered the
  mass mention and notified members.
- `kind: "suppressed_text"` — the content contains `@everyone`/`@here` but
  `mention_everyone` is false: the author lacked the MENTION_EVERYONE permission, so the
  text is rendered as-is and **no notification is sent**.

Each hit also carries the author's current permission context (`authorHasMentionEveryoneNow`,
`authorGrantingRoles`, `channelAllowsEveryone`). `read_messages` and the forum message
serializer expose the same distinction as `mentionEveryone` / `mentions` / `roleMentionIds`.

## Feature audit

`docs/analysis/FEATURE_AUDIT.md` lists all 115 tools with their confirmation model and the verification
evidence gathered against a live server (live API run vs unit test), plus the defects that audit fixed.

## 115-tool contract status

The canonical registry target in this branch: **115 canonical tools** (24 baseline + 86 expansion + 2 permission introspection + 1 mass-mention audit + 2 member admin). All tools are covered by registry-count, router-coverage, and runtime-contract tests. See `tests/test_tool_runtime_contracts.py` for the detailed contract assertions.
