# discord.py coverage gaps (list only — nothing implemented)

Method: introspected the installed **discord.py 2.7.1** public async API
(`Client`, `Guild`, `abc.GuildChannel`, `abc.Messageable`, channel types, `Thread`, `Message`,
`Webhook`, `Emoji`, `Sticker`, `Invite`, `AutoModRule`, `ScheduledEvent`, `StageInstance`,
`Template`, `VoiceClient`, `SoundboardSound`, `Entitlement`) and diffed it against the **115 MCP
tools** in this repo; then confirmed each absence with a grep sweep over `src/discord_mcp`.

Legend: **covered** = an existing tool does it · **missing** = no tool · **N/A** = not meaningful
for an MCP server (bot-framework or client-only).

---

## 1. Covered today

| Domain | MCP tools |
|---|---|
| Guild settings | `update_guild` (full `Guild.edit`: name, description, locale, verification, content filter, notifications, mfa, community, discoverable, invites/widget/premium/raid booleans, afk, system/rules/public-updates/safety-alerts/widget channels, afk_timeout, system_channel_flags, owner, icon/banner/splash/discovery_splash images, invites/dms_disabled_until), `get_server_info`, `get_guild_vanity_url` |
| Channels | `get_channels*`, `get_channel_hierarchy`, `topology_*`, `create_text_channel`, `create_voice_channel`, `create_forum_channel`, `update_text_channel`, `update_voice_channel`, `update_forum_channel`, `delete_channel`, `create_category`, `rename_category`, `move_category`, `delete_category`, `set_channel_permission_overwrite`, `remove_channel_permission_overwrite`, `get_permission_overwrites`, `diff_channel_permissions`, `get_channel_type_counts`, `list_inactive_channels` |
| Messages | `send_message` (text), `send_embed_message`, `send_rich_announcement`, `read_messages`, `edit_message`, `reply_message`, `moderate_message`, `moderation_bulk_delete`, `add_reaction`, `add_multiple_reactions`, `remove_reaction`, `crosspost_announcement`, `download_attachment`, `audit_mass_mentions` |
| Members | `list_members`, `get_user_info`, `set_member_nickname`, `set_member_roles`, `add_role`, `remove_role`, `add_roles_bulk`, `remove_roles_bulk`, `mute_member_role_based`, `unmute_member_role_based`, `moderation_kick_member`, `moderation_ban_member`, `moderation_timeout_member`, `remove_member_timeout`, `bulk_ban_members`, `prune_inactive_members`, `unban_member`, `dynamic_role_provision`, `compute_member_permissions`, `get_role_permissions` |
| Roles | `create_role`, `update_role`, `delete_role`, `get_role_hierarchy`, `topology_role_hierarchy`, `permission_drift_check`, `export_server_snapshot` |
| Forums/threads | `read_forum_threads`, `list_threads`, `search_threads`, `list_forum_posts`, `read_forum_post_messages`, `read_forum_posts_batch`, `get_thread_context`, `list_thread_participants`, `get_thread_activity_summary`, `tag_forum_post`, `retag_forum_post`, `add_thread_tags`, `unarchive_thread` |
| Audit | `get_audit_log`, `get_member_moderation_history`, `get_channel_activity_summary`, `get_incident_timeline`, `get_audit_actor_summary`, `check_audit_reason_compliance`, `server_health_check`, `governance_evidence_packager` |
| AutoMod | `automod_get_ruleset`, `automod_apply_ruleset`, `automod_validate_ruleset`, `list_auto_moderation_rules`, `create_auto_moderation_rule`, `update_auto_moderation_rule`, `automod_export_rules`, `automod_rollback_ruleset` |
| Onboarding/welcome | `get_guild_onboarding`, `update_guild_onboarding`, `get_guild_welcome_screen`, `update_guild_welcome_screen` |
| Webhooks | `create_channel_webhook`, `list_channel_webhooks`, `execute_channel_webhook` |
| Integrations | `list_guild_integrations` |
| Incidents (MCP-side) | `incident_get_channel_state`, `incident_set_channel_state`, `incident_apply_lockdown`, `incident_rollback_lockdown`, `create_incident_room`, `append_incident_event`, `close_incident` |

---

## 2. Missing (grouped, with the discord.py call that would back each)

### 2.1 Invites & membership discovery
| Capability | discord.py | Notes |
|---|---|---|
| Create an invite | `abc.GuildChannel.create_invite(max_age, max_uses, temporary, unique, target_type)` | high value for an admin MCP |
| List a guild's/channel's invites | `Guild.invites()`, `abc.GuildChannel.invites()` | |
| Delete an invite | `Invite.delete()` | destructive → needs `dry_run`/`confirm_token` |
| Ban list | `Guild.fetch_bans()`, `Guild.fetch_ban(user)` | `fetch_ban` is used internally by `unban_member` only |
| Member search by prefix | `Guild.query_members(query=..., limit=...)` | needs `Intents.members` |
| Role member counts | `Guild.role_member_counts()` | |
| Prune estimate | `Guild.estimate_pruned_members(days)` | pairs with `prune_inactive_members` |

### 2.2 Messages
| Capability | discord.py | Notes |
|---|---|---|
| Upload files/attachments | `abc.Messageable.send(file=..., files=...)` | `send_message` is text-only today |
| Stickers / TTS / silent / nonce / allowed_mentions / suppress_embeds | same `send(...)` params | |
| Buttons, selects, action rows | `send(view=...)`, `discord.ui.View` | needs component builders |
| Polls (create) | `send(poll=discord.Poll(...))` (`discord.Poll` exists in 2.7) | read side also missing |
| Forward a message | `Message.forward(destination)` | |
| Pin / unpin | `Message.pin()`, `unpin()` | |
| Clear all reactions | `Message.clear_reactions()` | only single-emoji removal exists |
| Who reacted | `Reaction.users()` | |
| Thread from a message | `Message.create_thread(name, ...)` | |
| Typing indicator | `abc.Messageable.typing()` | mostly cosmetic |

### 2.3 Threads (outside the forum read/tag surface)
| Capability | discord.py |
|---|---|
| Create a thread in a text channel | `TextChannel.create_thread(name, type, ...)` |
| Join / leave a thread | `Thread.join()`, `Thread.leave()` |
| Add / remove a thread member | `Thread.add_user()`, `Thread.remove_user()` |
| Rename / lock / slowmode / archive a thread | `Thread.edit(...)` (only `unarchive_thread` exists) |
| Delete a thread | `Thread.delete()` |
| Active threads of a guild | `Guild.active_threads()` |

### 2.4 Channels
| Capability | discord.py | Notes |
|---|---|---|
| Clone any channel | `GuildChannel.clone()`, `CategoryChannel.clone()` | |
| Announcement (news) channel | `Guild.create_text_channel(news=True)` | create tools cover text/voice/forum only |
| Stage channel | `Guild.create_stage_channel()` | |
| Channel following (announcement follow) | `TextChannel.follow(webhook_channel)` | |
| Permission sync to children | (client behaviour; no single API call) | would be an MCP-side helper over overwrites |

### 2.5 Members, voice states, DMs
| Capability | discord.py | Notes |
|---|---|---|
| Server mute / deafen a member | `Guild.change_voice_state(member, mute=, deafen=)` | |
| Move a member between voice channels | `Member.move_to(channel)` | |
| Kick a member from voice | `Member.move_to(None)` | |
| Request to speak (stage) | `Member.request_to_speak()` | |
| Voice state read | `Member.voice`, `Member.fetch_voice()` | |
| Per-member flags | `Member.edit(flags=..., bypass_verification=...)` | |
| Member avatar / banner / bio | `Member.avatar`, `Member.banner`, `Member.edit(banner=...)` | |
| DM a user / create DM channel | `User.create_dm()`, `Member.create_dm()` | privacy-sensitive |
| Bot's own profile | `Client.user.edit(username=, avatar=, banner=)`, `Client.edit(...)` | |

### 2.6 Roles
| Capability | discord.py |
|---|---|
| Role icon (display icon) | `Role.edit(display_icon=...)` |
| Bulk reorder roles | `Guild.edit_role_positions(positions={role: position})` |
| Role tags (bot/integration metadata) | `Role.tags` (read) |
| Role member counts | `Guild.role_member_counts()` |

### 2.7 Emoji, stickers, soundboard
| Capability | discord.py |
|---|---|
| Guild emoji: create / delete / edit / fetch | `Guild.create_custom_emoji()`, `delete_emoji()`, `Emoji.edit()`, `Guild.fetch_emojis()` |
| Application emoji CRUD (2.5+) | `Client.create_application_emoji()`, `fetch_application_emoji(s)`, delete |
| Guild sticker: create / delete / fetch | `Guild.create_sticker()`, `delete_sticker()`, `fetch_stickers()` |
| Soundboard sounds: create / fetch / edit / delete / play | `Guild.create_soundboard_sound()`, `fetch_soundboard_sounds()`, `SoundboardSound.edit()/delete()`, `VoiceChannel.send_sound()` |

### 2.8 Webhooks
| Capability | discord.py |
|---|---|
| Edit / delete a webhook | `Webhook.edit()`, `Webhook.delete()` |
| Webhook messages: fetch / edit / delete | `Webhook.fetch_message()`, `edit_message()`, `delete_message()` |

### 2.9 Scheduled events, stage, templates, widget, discovery
| Capability | discord.py |
|---|---|
| Scheduled events CRUD + lifecycle | `Guild.create_scheduled_event()`, `fetch_scheduled_event(s)`, `ScheduledEvent.edit()/delete()/start()/end()/cancel()` |
| Event subscribers (RSVP) | `ScheduledEvent.fetch_users()` |
| Stage instances | `StageChannel.create_instance()`, `fetch_instance()`, `StageInstance.edit()/delete()` |
| Guild templates | `Guild.create_template()`, `templates()`, `Template.sync()/edit()/delete()`, `Client.fetch_template()` |
| Guild preview | `Client.fetch_guild_preview()` |
| Widget settings read / dedicated edit | `Guild.widget()`, `Guild.edit_widget()` (only the `Guild.edit` fields are exposed today) |
| Server discovery data | `Guild.discovery_splash` (set ✓), discovery listing = N/A |

### 2.10 Monetization
| Capability | discord.py |
|---|---|
| SKUs / entitlements | `Client.fetch_skus()`, `Client.fetch_entitlement(s)`, `Entitlement.consume()/delete()`, `Client.create_entitlement()` |

### 2.11 Interactions / application commands
| Capability | discord.py | Notes |
|---|---|---|
| Slash/context command registration + sync | `Client.tree.sync()`, `app_commands` | N/A-ish: the MCP exposes tools over MCP, not Discord slash commands |
| Component/modal responses, ephemeral replies | `Interaction.response`, `discord.ui` | needs a gateway interaction loop |

---

## 3. Not applicable

- **Bot-framework plumbing** (no Discord feature behind it): `commands.Bot` prefix commands, cogs,
  `tasks.loop`, view persistence, sharding, `Guild.chunk()`, `Guild.leave()`, `Client.login/start/close`,
  gateway event handlers, raw events, `Intents` tuning.
- **Client-only Discord features** (no public API — verified against the guild resource and the API
  changelog): Server Profile **banner colour**, **traits**, **games**, **private profile**, and the
  **server tag** (the tag lives on the *user* as `primary_guild`). `update_guild` rejects them as
  `unsupported_fields`.
- **Voice transport**: `VoiceClient.connect/disconnect/move_to`, voice receive — needs a live voice
  gateway connection, which an stdio MCP server has no business holding.
- `Guild.membership_screening` / `edit_membership_screening` and `VoiceChannel.edit(status=...)` do not
  exist in discord.py 2.7.1 (checked), so there is nothing to align.

---

## 4. Suggested order if/when we implement

1. **Invites + ban list + member search** (`create_invite`, `list_invites`, `delete_invite`,
   `list_bans`, `search_members`) — daily admin work, low risk (delete needs the confirm gate).
2. **Message completeness**: file upload, pin/unpin, forward, clear_reactions, poll create.
3. **Thread management**: create/join/leave/add member/edit/delete + `active_threads`.
4. **Emoji/sticker CRUD** and **webhook edit/delete** (small, self-contained).
5. **Scheduled events + stage** (feature-rich, needs more schema work).
6. **Voice states** (`change_voice_state`, `move_to`) — needs `Intents.voice_states`.
7. **Guild templates, widget read/edit, guild preview, role icon/reorder, soundboard**.
8. **Entitlements/SKUs** — only for monetized apps; likely skip.
