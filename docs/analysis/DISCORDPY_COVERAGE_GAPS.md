# discord.py coverage gaps — closed (registry = 212 tools)

Method: introspected the installed **discord.py 2.7.1** public async API and produced the
original gap list; the canonical per-tool contract now lives in
`docs/product/coverage-gaps-implementation.md`. Registry: **212** MCP tools (was 116; 96 added).
Every name below verified against `compose_tool_registry()`.

Legend: **Implemented** = dedicated tool · **Partly** = covered via a broader handler · **N/A** =
not meaningful for an MCP server (bot-framework plumbing, client-only fields, voice transport).

---

## 1. Covered today

| Domain | MCP tools (new in this pass) | Status |
|---|---|---|
| Invites & bans | `create_invite`, `list_invites`, `delete_invite`, `list_bans`, `get_ban`, `search_members`, `get_role_member_counts`, `estimate_pruned_members` | Implemented |
| Messages (advanced) | `send_message_with_files`, `send_components`, `send_poll`, `get_poll_results`, `forward_message`, `pin_message`, `unpin_message`, `clear_message_reactions`, `get_reaction_users`, `create_thread_from_message`, `send_typing` | Implemented |
| Threads | `create_thread`, `join_thread`, `leave_thread`, `add_thread_member`, `remove_thread_member`, `edit_thread`, `delete_thread`, `list_active_threads` | Implemented |
| Channels | `clone_channel`, `create_announcement_channel`, `create_stage_channel`, `follow_channel`, `sync_channel_permissions`, `set_voice_channel_status` | Implemented |
| Members / voice / DMs | `change_member_voice_state`, `move_member_voice`, `request_to_speak`, `get_member_voice_state`, `edit_member_profile`, `create_dm_channel`, `update_bot_profile`, `set_role_icon`, `reorder_roles`, `get_role_details` | Implemented / Partly |
| Emoji / sticker / soundboard | `create_emoji`, `edit_emoji`, `delete_emoji`, `create_application_emoji`, `edit_application_emoji`, `delete_application_emoji`, `list_application_emojis`, `create_sticker`, `edit_sticker`, `delete_sticker`, `list_stickers`, `create_soundboard_sound`, `list_soundboard_sounds`, `edit_soundboard_sound`, `delete_soundboard_sound`, `send_soundboard_sound` | Implemented |
| Webhooks | `get_webhook`, `edit_webhook`, `delete_webhook`, `get_webhook_message`, `edit_webhook_message`, `delete_webhook_message` | Implemented |
| Scheduled events / stage / templates / widget / discovery | 13 scheduled-event tools + 4 stage-instance + 7 template/widget | Implemented |
| Monetization | `list_skus`, `list_entitlements`, `get_entitlement`, `create_entitlement`, `consume_entitlement`, `delete_entitlement` | Implemented (app-cmd surface still out of scope) |
| Interactions | `list_app_commands`, `get_app_command`, `sync_app_commands` | Implemented (registration via introspection only) |

---

## 2. Gaps — now implemented (per domain)

Table format now: **Capability | discord.py backing | Status | Tool(s)**.

### 2.1 Invites & membership discovery → all Implemented

| Capability | discord.py | Status | Tool |
|---|---|---|---|
| Invite creation | `abc.GuildChannel.create_invite` | Implemented | `create_invite` |
| Invite list / delete | `Guild.invites()` / `Invite.delete()` (revoke missing) | Implemented | `list_invites`, `delete_invite` |
| Ban list / fetch | `Guild.bans()` (was `fetch_bans`) | Implemented | `list_bans`, `get_ban` |
| Member search / counts / prune | `Guild.members`, `Role.members`, `search_members` | Implemented | `search_members`, `get_role_member_counts`, `estimate_pruned_members` |

### 2.2 Messages (advanced) → Implemented / Partly

`send_message_with_files` (files, stickers, TTS, silent, allowed-mentions); `send_components`; `send_poll` / `get_poll_results`; `forward_message`, `pin_message` / `unpin_message`, `clear_message_reactions`, `get_reaction_users`, `create_thread_from_message`, `send_typing`.

Ephemeral / component responses: **out of scope** (see §4).

### 2.3 Threads → Implemented
`create_thread`, `join_thread`, `leave_thread`, `add_thread_member`, `remove_thread_member`, `edit_thread`, `delete_thread`, `list_active_threads`, `create_thread_from_message`. Note: `discord.ThreadType` does not exist; thread kinds live on `discord.ChannelType`.

### 2.4 Channels → Implemented (`set_voice_channel_status` replaces `VoiceChannel.edit(status=...)`)
`clone_channel`, `create_announcement_channel`, `create_stage_channel`, `follow_channel`, `sync_channel_permissions`, `set_voice_channel_status`. `VoiceChannel.edit(status=...)` is backed by `abc.py` → `edit_voice_channel_status`; original analysis wrongly claimed it absent.

### 2.5 Members, voice states, DMs → Implemented / Partly
`change_member_voice_state`, `move_member_voice`, `request_to_speak`, `get_member_voice_state`, `edit_member_profile`, `create_dm_channel`, `update_bot_profile`. `Guild.change_voice_state` only changes the bot's own gateway opcode-4 state; per-member voice uses `Member.edit` (mute/deafen). `Client.edit()` does not exist — bot profile goes through `ClientUser.edit`.

### 2.6 Roles → Implemented (`set_role_icon`, `reorder_roles`, `get_role_details`)
`Member.add_role`/`remove_role` do not exist in 2.7.1; they are `add_roles` / `remove_roles`. `Role.is_integration_managed` is `is_integration()`.

### 2.7 Emoji, stickers, soundboard → Implemented
`create_emoji`, `edit_emoji`, `delete_emoji` (was `Guild.delete_custom_emoji`); application-level emoji tools; `create_sticker`, `edit_sticker`, `delete_sticker` (the original analysis wrongly listed sticker editing as missing — `Sticker.edit()`/`delete()` exist); `create_soundboard_sound`, `edit_soundboard_sound`, `delete_soundboard_sound`, `list_soundboard_sounds`, `send_soundboard_sound`.

### 2.8 Webhooks → Implemented (`get_webhook`, `edit_webhook`, `delete_webhook`, `get_webhook_message`, `edit_webhook_message`, `delete_webhook_message`)

### 2.9 Scheduled events / stage / templates / widget / discovery → Implemented
| Capability | discord.py | Status / Tools |
|---|---|---|
| Scheduled events (create / list / fetch / edit / delete / start / end / cancel) | `Guild.create_scheduled_event`, `scheduled_events`, `fetch_scheduled_event(s)` (`with_counts`) | Implemented | `create_scheduled_event`, `get_scheduled_event`, `list_scheduled_events`, `edit_scheduled_event`, `delete_scheduled_event`, `start_scheduled_event`, `end_scheduled_event`, `cancel_scheduled_event` |
| Scheduled event users | `ScheduledEvent.users()` (no `with_member`) | Partly | `list_scheduled_event_users` |
| Stage instances (create / get / edit / delete) | `StageChannel.create_instance()` | Implemented | `create_stage_instance`, `get_stage_instance`, `edit_stage_instance`, `delete_stage_instance` |
| Guild templates (list / create / get / sync / edit / delete) | `Guild.templates()` / `Template.*` | Implemented | `list_templates`, `create_template`, `get_template`, `sync_template`, `edit_template`, `delete_template` |
| Widget settings | `Guild.widget` | Implemented | `get_widget_settings`, `edit_widget_settings` |
| Guild preview | `Guild.preview()` | Implemented | `get_guild_preview` |

Notes: `Guild.fetch_scheduled_event(s)` take `with_counts`, not `with_user_count` — the tools expose `with_user_count` and map it. `ScheduledEvent.users()` has no `with_member` parameter, so `list_scheduled_event_users` returns users without member expansion.

### 2.10 Monetization → Implemented (app-cmd surface out of scope)
`list_skus`, `list_entitlements`, `get_entitlement`, `create_entitlement`, `consume_entitlement`, `delete_entitlement`. `Client.fetch_entitlements()` is `entitlements()`.

### 2.11 Interactions / application commands → Implemented (introspection only)
`list_app_commands`, `get_app_command`, `sync_app_commands`. Live slash-registration is out of scope.

---

## 3. Corrections to this analysis (verified against discord.py 2.7.1)

Every claim below can be re-checked against the installed runtime (`import discord`; `inspect(signature(...))` / `dir(...)`).

- **§3 claim `VoiceChannel.edit(status=...)` missing** — false. It exists via `**options` → `abc.py` → `http.edit_voice_channel_status`; implemented as `set_voice_channel_status`.
- **Sticker editing** — `GuildSticker.edit()` / `delete()` exist; original listed it missing. Now `edit_sticker`.
- `Guild.fetch_bans()` → `Guild.bans()`; `Member.add_role` / `remove_role` → `add_roles` / `remove_roles`; `Invite.revoke` does not exist; `Sticker.edit` / `delete` do exist; `Member.mute` / `deafen` do not exist (use `Member.edit`); `ScheduledEvent.fetch_users()` → `.users()`; `Client.fetch_entitlements()` → `.entitlements()`; `Role.is_integration_managed` → `.is_integration()`; `Guild.delete_custom_emoji` → `delete_emoji`; `Guild.create_stage_instance` → `StageChannel.create_instance()`; `Guild.create_invite` → `abc.GuildChannel.create_invite`.
- `Guild.change_voice_state` only mutates the bot's own gateway opcode-4 state; member-level voice uses `Member.edit`.
- `Guild.fetch_scheduled_event(s)` accept `with_counts`, not `with_user_count`; `ScheduledEvent.users` takes no `with_member`.
- `discord.Permissions` is a `BaseFlags`, not `IntFlag` — `int(perms)` raises `TypeError`.
- `discord.ThreadType` does not exist; thread kinds are on `discord.ChannelType`.

---

## 4. Deliberately out of scope (post-implementation)

- **Slash-command live registration** beyond introspection (`list_app_commands` / `get_app_command`) — requires app-registration flow, not bot runtime.
- **Ephemeral / interactive component responses** (buttons/selects with follow-up) — needs an interaction endpoint the MCP server does not expose.
- **Voice transport** (receiver, playback, voice-state sync) — bot-framework plumbing; verified N/A.
- **Monetization enforcement against a non-monetized app** — `entitlements()` / SKU tools exist but have no real data source.
- **Bot-framework plumbing** (`commands.Bot`, cogs) and **client-only server-profile fields** (`Client.edit` missing; bot profile uses `ClientUser.edit`) remain N/A.
- `Guild.membership_screening` does not exist in 2.7.1; kept out.
