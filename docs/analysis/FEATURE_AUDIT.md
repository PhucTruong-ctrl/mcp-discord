# Feature audit - all 113 MCP tools

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
| 24 | `update_guild` | Guild settings | direct | live | closes the description/verification-level gap |
| 25 | `create_voice_channel` | Channels | direct | unit test |  |
| 26 | `create_forum_channel` | Channels | direct | unit test |  |
| 27 | `update_text_channel` | Channels | direct | unit test |  |
| 28 | `update_voice_channel` | Channels | direct | unit test |  |
| 29 | `update_forum_channel` | Channels | direct | unit test |  |
| 30 | `list_forum_posts` | Forum intel | direct | live |  |
| 31 | `read_forum_post_messages` | Forum intel | direct | unit test |  |
| 32 | `read_forum_posts_batch` | Forum intel | direct | unit test |  |
| 33 | `get_thread_context` | Forum intel | direct | unit test |  |
| 34 | `list_thread_participants` | Forum intel | direct | unit test |  |
| 35 | `get_thread_activity_summary` | Forum intel | direct | live |  |
| 36 | `tag_forum_post` | Forum intel | direct | unit test |  |
| 37 | `retag_forum_post` | Forum intel | direct | unit test |  |
| 38 | `get_channels_structured` | Inventory/permissions | direct | live |  |
| 39 | `get_channel_hierarchy` | Inventory/permissions | direct | live |  |
| 40 | `get_role_hierarchy` | Inventory/permissions | direct | live |  |
| 41 | `get_permission_overwrites` | Inventory/permissions | direct | live |  |
| 42 | `diff_channel_permissions` | Inventory/permissions | direct | live |  |
| 43 | `export_server_snapshot` | Inventory/permissions | direct | live |  |
| 44 | `get_channel_type_counts` | Inventory/permissions | direct | live |  |
| 45 | `list_inactive_channels` | Inventory/permissions | direct | live |  |
| 46 | `set_channel_permission_overwrite` | Channel overwrites | direct | live | closes the overwrite-write gap (parameter object required by discord.py) |
| 47 | `remove_channel_permission_overwrite` | Channel overwrites | direct | live | deletes an explicit overwrite only |
| 48 | `moderation_bulk_delete` | Moderation core | dry_run + confirm_token | live |  |
| 49 | `moderation_timeout_member` | Moderation core | dry_run + confirm_token | unit test | execute needs a real member; gateway call unit-tested (member.timeout) |
| 50 | `moderation_kick_member` | Moderation core | dry_run + confirm_token | unit test | execute needs a real member; gateway call unit-tested (member.kick) |
| 51 | `moderation_ban_member` | Moderation core | dry_run + confirm_token | unit test | execute needs a real member; gateway call unit-tested (guild.ban) |
| 52 | `topology_channel_tree` | Topology | direct | live |  |
| 53 | `topology_channel_children` | Topology | direct | live |  |
| 54 | `topology_role_hierarchy` | Topology | direct | live |  |
| 55 | `topology_permission_matrix` | Topology | direct | live |  |
| 56 | `create_role` | Role governance | direct | live |  |
| 57 | `delete_role` | Role governance | direct | live |  |
| 58 | `update_role` | Role governance | direct | live |  |
| 59 | `add_roles_bulk` | Role governance | dry_run + confirm_token | live |  |
| 60 | `remove_roles_bulk` | Role governance | dry_run + confirm_token | live |  |
| 61 | `mute_member_role_based` | Role governance | direct | live |  |
| 62 | `unmute_member_role_based` | Role governance | direct | live |  |
| 63 | `permission_drift_check` | Role governance | direct | live |  |
| 64 | `get_audit_log` | Audit analytics | direct | live |  |
| 65 | `get_member_moderation_history` | Audit analytics | direct | live |  |
| 66 | `get_channel_activity_summary` | Audit analytics | direct | live |  |
| 67 | `get_incident_timeline` | Audit analytics | direct | live |  |
| 68 | `get_audit_actor_summary` | Audit analytics | direct | live |  |
| 69 | `check_audit_reason_compliance` | Audit analytics | direct | live |  |
| 70 | `server_health_check` | Audit analytics | direct | live |  |
| 71 | `governance_evidence_packager` | Audit analytics | direct | live |  |
| 72 | `get_guild_welcome_screen` | Onboarding | direct | unit test |  |
| 73 | `update_guild_welcome_screen` | Onboarding | direct | unit test |  |
| 74 | `get_guild_onboarding` | Onboarding | direct | live |  |
| 75 | `update_guild_onboarding` | Onboarding | direct | unit test |  |
| 76 | `dynamic_role_provision` | Onboarding | direct | unit test |  |
| 77 | `verification_gate_orchestrator` | Onboarding | local only | unit test |  |
| 78 | `progressive_access_unlock` | Onboarding | local only | unit test |  |
| 79 | `onboarding_friction_audit` | Onboarding | local only | unit test |  |
| 80 | `send_embed_message` | Messaging/workflow | direct | unit test |  |
| 81 | `send_rich_announcement` | Messaging/workflow | direct | unit test |  |
| 82 | `crosspost_announcement` | Messaging/workflow | direct | unit test |  |
| 83 | `create_channel_webhook` | Messaging/workflow | direct | unit test |  |
| 84 | `list_channel_webhooks` | Messaging/workflow | direct | live |  |
| 85 | `execute_channel_webhook` | Messaging/workflow | direct | unit test |  |
| 86 | `list_guild_integrations` | Messaging/workflow | direct | live |  |
| 87 | `get_guild_vanity_url` | Messaging/workflow | direct | live |  |
| 88 | `incident_get_channel_state` | Incident ops | local only | live |  |
| 89 | `incident_set_channel_state` | Incident ops | local only | live |  |
| 90 | `incident_apply_lockdown` | Incident ops | dry_run + confirm_token | live |  |
| 91 | `incident_rollback_lockdown` | Incident ops | dry_run + confirm_token | live |  |
| 92 | `automod_validate_ruleset` | AutoMod policy | local only | unit test |  |
| 93 | `automod_get_ruleset` | AutoMod policy | direct | live |  |
| 94 | `automod_apply_ruleset` | AutoMod policy | dry_run + confirm_token | live |  |
| 95 | `automod_rollback_ruleset` | AutoMod policy | dry_run + confirm_token | unit test | Discord has no rollback API: returns not_supported by design |
| 96 | `remove_member_timeout` | Expansion utilities | direct | live(api reached) |  |
| 97 | `unban_member` | Expansion utilities | direct | live(error path) |  |
| 98 | `bulk_ban_members` | Expansion utilities | dry_run + confirm_token | unit test | would ban real users; guild.bulk_ban unit-tested + dry-run verified live |
| 99 | `prune_inactive_members` | Expansion utilities | dry_run + confirm_token | unit test | would prune real users; guild.prune_members unit-tested + dry-run verified live |
| 100 | `create_category` | Expansion utilities | direct | live |  |
| 101 | `rename_category` | Expansion utilities | direct | live |  |
| 102 | `move_category` | Expansion utilities | direct | live |  |
| 103 | `delete_category` | Expansion utilities | dry_run + confirm_token | live |  |
| 104 | `create_incident_room` | Expansion utilities | direct | live |  |
| 105 | `append_incident_event` | Expansion utilities | direct | live |  |
| 106 | `close_incident` | Expansion utilities | direct | live |  |
| 107 | `list_auto_moderation_rules` | Expansion utilities | direct | live |  |
| 108 | `create_auto_moderation_rule` | Expansion utilities | direct | unit test |  |
| 109 | `update_auto_moderation_rule` | Expansion utilities | direct | unit test |  |
| 110 | `automod_export_rules` | Expansion utilities | direct | live |  |
| 111 | `get_role_permissions` | Permission intel | direct | live |  |
| 112 | `compute_member_permissions` | Permission intel | direct | live |  |
| 113 | `audit_mass_mentions` | Mass mentions | direct | live |  |
| 114 | `set_member_roles` | Member admin | dry_run + confirm_token | live |  |
| 116 | `list_guild_emojis` | Emoji | direct | live |  |

## Findings fixed in this audit

- Emoji round trips: the welcome screen dropped custom emoji ids, forum tags / `default_reaction_emoji`
  were write-only, onboarding `defaultChannels` returned names where the write path needs snowflakes, and
  nothing could list guild emojis. All emoji-bearing fields now share `core/emoji.py` (read
  `{emoji, emojiId, emojiAnimated}`; write unicode | `"name"` | `"<:name:id>"` | `{id, name, animated}`,
  bare names resolved against `guild.emojis`), `get_channels_structured` exposes `availableTags` /
  `defaultReactionEmoji`, and `list_guild_emojis` (tool 116) returns the guild's custom emoji ids.

- `get_server_info` read guild fields from the gateway cache, which is built from a
  truncated `GUILD_CREATE` for large guilds, so it reported `description: None` for a
  guild whose description was set. It now fetches the guild fresh and also reports
  `verification_level` (falling back to `approximate_member_count` when `member_count`
  is absent from the fetch payload).

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
- `update_guild` covered only 3 of the ~25 fields `Guild.edit` supports. It now maps the whole surface
  (text/enums/booleans/channels/timeouts/flags/owner/images/timestamps), loads images from a URL, data
  URI or path (`core/images.py`), rejects unknown fields and explains the 403 (MANAGE_GUILD, COMMUNITY
  needs ADMINISTRATOR, image fields need their guild feature). Server-profile banner colour, traits,
  games, private profile and the server tag have no public API and are rejected as unsupported.
- `set_member_roles` (tool 114) added: replaces a member's whole role set in one gated call, ignoring
  @everyone, rejecting integration-managed roles when *assigning* and auto-preserving them when
  replacing (Discord answers 403 50013 if the request drops one - that is what made a self-edit fail).
  `update_role` / `delete_role` also take `role_name` (unique, case-insensitive) and a null colour clears
  the primary colour or a gradient stop.
- Follow-up round-trip gaps found while testing the emoji codec: a read onboarding payload carries
  `"mode": null`, which the writer rejected (now treated as "not provided"); the welcome-screen writer
  only accepted `welcome_channels`/`channel_id` while the reader emits `welcomeChannels`/`channelId`
  (both accepted now); and `discord.WelcomeChannel.to_dict()` only fills `emoji_id` for an emoji
  *object*, so a `<:name:id>` string silently lost the id - the welcome path now passes a
  `PartialEmoji`. `tests/test_channel_admin_tools.py` also stubbed only part of the MCP SDK, so running
  that file alone failed with `No module named 'mcp.server.models'`; the stub now covers
  `mcp.server.models` and the `mcp.types` names `discord_mcp.server` imports.
- Role colours were write-only: `create_role`/`update_role` accepted `color` but no role reader
  returned it. `role_payload` now exposes `color`/`colorHex`/`secondaryColor`/`tertiaryColor`/
  `gradient`, gradient writes are supported (and their `670006` refusal explained), and
  `add_role`/`remove_role` no longer crash on an uncached role, accept a `reason`, and report
  before/after state from a fresh API read instead of a stale local cache.
- `update_guild_onboarding` crashed before any API call (`'dict' object has no attribute 'to_dict'`)
  because raw JSON prompts were passed to `Guild.edit_onboarding`, which calls `prompt.to_dict(id=i)`;
  it also sent partial PUTs. It now builds real `OnboardingPrompt` objects and merges onto the current
  configuration, and explains Discord's onboarding requirements (>= 7 public channels, >= 5 writable by
  `@everyone`) instead of surfacing a bare `400 350001`.
- `set_member_nickname` (tool 114) was added: real `Member.edit(nick=...)` with nickname removal via an
  empty string/null, 32-character validation, and a Forbidden error that names the required permission
  (`MANAGE_NICKNAMES`, or `CHANGE_NICKNAME` for the bot's own record). Verified live by renaming the bot
  and clearing it again.

## Not run against real members (by design)

Ban / kick / timeout / prune execute paths need a real victim; the gateway calls are unit-tested and
the dry-run + confirm_token gates were exercised live. `automod_rollback_ruleset` returns
`not_supported` because Discord exposes no rollback primitive.
