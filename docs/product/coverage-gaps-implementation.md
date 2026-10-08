# Coverage-gap implementation spec (new tools)

Authoritative contract for closing every gap in
[`docs/analysis/DISCORDPY_COVERAGE_GAPS.md`](../analysis/DISCORDPY_COVERAGE_GAPS.md).

**Every discord.py call below was verified against the installed discord.py 2.7.1**
(introspection, not docs). Where the gap document named a method that does **not** exist
in 2.7.1, the verified replacement is listed instead.

## Verified corrections to the gap document

| Gap doc claims | Reality in discord.py 2.7.1 |
|---|---|
| `Guild.fetch_bans()` | `Guild.bans(*, limit=1000, before, after)` — async iterator of `BanEntry` |
| `Member.add_role` / `remove_role` | `Member.add_roles` / `remove_roles` (plural) |
| `Member.mute()` / `deafen()` | **Do not exist.** Use `Member.edit(mute=, deafen=)` |
| `Sticker.edit()` / `Sticker.delete()` | **Do not exist on the base `Sticker`, but `GuildSticker` has both** — so the gap document omitted sticker editing entirely |
| `Guild.change_voice_state(member, ...)` | **Only changes the bot's own voice state** (gateway opcode 4); use `Member.edit(voice_channel=, mute=, deafen=)` for a member |
| `Invite.revoke()` | **Does not exist.** Use `Invite.delete()` |
| `ScheduledEvent.fetch_users()` | `ScheduledEvent.users(*, limit, before, after, oldest_first)` |
| `Client.fetch_entitlements()` | `Client.entitlements(...)` — async iterator |
| `Role.is_integration_managed` | `Role.is_integration()` |
| `Client.edit(...)` (bot profile) | **Does not exist in 2.7.1.** Use `Client.user.edit(username=, avatar=, banner=)` |
| `Guild.create_stage_instance` | `StageChannel.create_instance(...)` / `StageChannel.fetch_instance()` |
| `Guild.create_invite` | `abc.GuildChannel.create_invite(...)` |
| `Guild.delete_custom_emoji` | `Guild.delete_emoji` |
| `Guild.membership_screening` | Does not exist in 2.7.1 (already N/A) |
| §3: `VoiceChannel.edit(status=...)` "does not exist" | **Works in 2.7.1** — `abc.GuildChannel._edit` pops `status` from `**options` (`abc.py:564`) and calls `http.edit_voice_channel_status` |

## Shared conventions (every new tool obeys these)

### Layout — one domain = two files

| Path | Exports |
|---|---|
| `src/discord_mcp/tools/schemas/<domain>.py` | `<DOMAIN>_TOOLS: List[Tool]` |
| `src/discord_mcp/tools/handlers/<domain>.py` | `handle_<tool_name>(arguments, deps) -> List[TextContent]` |
| `tests/test_<domain>_tools.py` | `unittest.IsolatedAsyncioTestCase` tests |

Registry wiring (`tools/schemas/__init__.py`) and router wiring
(`tools/handlers/router.py`) are owned by the integrator, **not** by domain agents.
Do not edit those two files, `composition.py`, `server.py`, `core/safety.py` or
`core/state.py`.

### Handler contract

```python
async def handle_x(arguments: Dict[str, Any], deps: Dict[str, Any]) -> List[TextContent]:
    gateway = require_gateway(deps, "x")      # core.common.require_gateway
    guild = await gateway.resolve_guild(arguments["server_id"])
    ...
    return json_text({...})                    # core.common.json_text
```

- `require_gateway(deps, tool)` raises `ValueError(f"gateway is required for {tool}")`
  when `deps.get("gateway")` is falsy. **Every new tool is gateway-dependent** — never
  return a synthetic/placeholder payload when the gateway is absent.
- `json_text(payload)` returns `[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]`.
  Use `indent=2` only where the neighbouring code in the same domain already does.
- Snowflakes go in and out as **strings**. Never return a raw `int` id in a payload.
- Missing entity → `raise ValueError(...)` naming the id and the server. Never return an
  empty success payload for a miss.
- Do **not** catch broad exceptions to synthesise responses. Let `discord.Forbidden` /
  `discord.NotFound` surface, except where the pattern below says to translate.

### Destructive / privileged gate

Any tool that **creates, mutates or deletes** something server-side MUST use the
two-step gate from `core/safety.py`:

```python
from discord_mcp.core.safety import build_dry_run_result, verify_confirm_token
from discord_mcp.core.validation import require_reason

action = "<tool_name>"
targets = {...}                      # MUST be identical on both paths, sorted deterministically
if bool(arguments.get("dry_run", True)):
    return json_text(build_dry_run_result(action, targets, {...}))
verify_confirm_token(action, targets, arguments.get("confirm_token"))
# ... perform the mutation ...
return json_text({"status": "executed", "action": action, ...})
```

Rules for the gate:
- `dry_run` **defaults to `True`** (dry-run is the default path).
- Read-only tools take **no** `dry_run`, `confirm_token` or `reason` parameters.
- Gate `reason` through `require_reason(arguments.get("reason"), action)` when the action
  is destructive (delete/ban/prune/cancel), otherwise pass it through as optional.
- `targets` values must be JSON-stable: `sorted()` every list, `str()` every id.

### Schema contract

- One `Tool(name=..., description=..., inputSchema={...})` per tool.
- `inputSchema` is `{"type": "object", "properties": {...}, "required": [...]}`.
- Every gate-taking tool declares `dry_run` (`bool`, default true), `confirm_token`
  (`string`) and, where required, `reason` (`string`) in `properties`.
- `description` states what it does **and** which layer decides the tricky part
  (permission, intent requirement, irreversibility). No marketing prose.

### Tests

- `tests/test_<domain>_tools.py`, `unittest.IsolatedAsyncioTestCase`, pytest-runnable.
- Cover **consumer-visible behaviour**, not wiring: payload shape and required keys,
  resolution/miss errors, gate enforcement (dry-run returns `confirmToken`; execute
  without a token raises `ValueError`), and argument validation.
- Fake the gateway with `unittest.mock.AsyncMock` / small local fakes, exactly like
  `tests/test_permission_intel_tools.py` and `tests/test_moderation_core_tools.py`.
- Do **not** assert the tool is registered in the router or in the registry — the
  integrator owns that contract and pins the totals.
- Do **not** import anything from `discord_mcp.tools.handlers.router` or
  `discord_mcp.tools.schemas` in your tests.

---

## Domain A1 — `invites_membership`

Files: `schemas/invites_membership.py` → `INVITES_MEMBERSHIP_TOOLS`;
`handlers/invites_membership.py`.

| Tool | Args (required first) | discord.py 2.7.1 | Gate |
|---|---|---|---|
| `create_invite` | `server_id`, `channel_id`; opt `max_age`, `max_uses`, `temporary`, `unique`, `target_type`, `target_user`, `target_application_id`, `guest`, `reason` | `channel.create_invite(max_age=0, max_uses=0, temporary=False, unique=True, target_type=None, target_user=None, target_application_id=None, guest=False, reason=None)` | yes |
| `list_invites` | `server_id`; opt `channel_id` | `guild.invites()` or `channel.invites()` | no |
| `delete_invite` | `server_id`, `invite_code`, `reason` | `Invite.delete()` | yes, reason required |
| `list_bans` | `server_id`; opt `limit` | `async for e in guild.bans(limit=...)` → `BanEntry(user, reason)` | no |
| `get_ban` | `server_id`, `user_id` | `guild.fetch_ban(user)` | no |
| `search_members` | `server_id`; opt `query`, `limit`, `user_ids` | `guild.query_members(query=..., limit=5, user_ids=...)` | no |
| `get_role_member_counts` | `server_id` | `guild.role_member_counts()` → `Dict[Role|Object, int]` | no |
| `estimate_pruned_members` | `server_id`, `days`; opt `role_ids` | `guild.estimate_pruned_members(days=..., roles=...)` | no |

Notes: `list_invites` returns both guild-wide and per-channel rows when `channel_id`
given; include `channelId` per row. `BanEntry` has `.user` (User|Object) and `.reason`.
`query_members` requires the **members** intent (already enabled in `server.py`) —
mention that in the schema description. `estimate_pruned_members` returns `Optional[int]`
(`None` means "no prune possible"); surface that as `{"prunable": bool, "count": int|None}`.

## Domain A2 — `thread_management`

Files: `schemas/thread_management.py` → `THREAD_MANAGEMENT_TOOLS`;
`handlers/thread_management.py`.

| Tool | Args | discord.py 2.7.1 | Gate |
|---|---|---|---|
| `create_thread` | `server_id`, `channel_id`, `name`; opt `auto_archive_duration`, `slowmode_delay`, `type`, `reason` | `TextChannel.create_thread(name, *, auto_archive_duration=..., slowmode_delay=None, reason=None)`; private/public via `ThreadType` | yes |
| `join_thread` | `server_id`, `thread_id` | `thread.join()` | no |
| `leave_thread` | `server_id`, `thread_id` | `thread.leave()` | no |
| `add_thread_member` | `server_id`, `thread_id`, `user_id` | `thread.add_user(user)` | yes |
| `remove_thread_member` | `server_id`, `thread_id`, `user_id` | `thread.remove_user(user)` | yes |
| `edit_thread` | `server_id`, `thread_id`; opt `name`, `archived`, `locked`, `invitable`, `pinned`, `slowmode_delay`, `auto_archive_duration`, `reason` | `thread.edit(...)` | yes |
| `delete_thread` | `server_id`, `thread_id`, `reason` | `thread.delete()` | yes, reason required |
| `list_active_threads` | `server_id` | `guild.active_threads()` → `List[Thread]` | no |

Notes: resolve threads with `gateway.resolve_thread(thread_id, server_id)` which returns
`(thread, guild)`. `create_thread` needs a **text** channel — reject a forum parent with a
clear `ValueError` telling the caller to use the forum-post tools. `discord.ThreadType`
does **not** exist in 2.7.1 — the thread types are members of `discord.ChannelType`
(`public_thread` / `private_thread`); parse the `type` arg against those names and
raise a clear error on an unknown value. Note `TextChannel.create_thread` defaults
`type=None` to a **private** thread.

## Domain A3 — `messages_advanced`

Files: `schemas/messages_advanced.py` → `MESSAGES_ADVANCED_TOOLS`;
`handlers/messages_advanced.py`.

| Tool | Args | discord.py 2.7.1 | Gate |
|---|---|---|---|
| `send_message_with_files` | `server_id`, `channel_id`; opt `content`, `file_paths` (list of local paths), `file_urls` (list of http(s) URLs), `sticker_ids`, `tts`, `silent`, `suppress_embeds`, `mention_everyone`, `mention_author`, `nonce`, `allowed_mention_roles`, `allowed_mention_users`, `allowed_mention_echo`, `delete_after` | `Messageable.send(content, tts, file, files, stickers, delete_after, nonce, allowed_mentions, mention_author, suppress_embeds, silent, poll)` | no |
| `send_components` | `server_id`, `channel_id`, `components` (list of component specs) | `send(view=discord.ui.View(...))` | no |
| `send_poll` | `server_id`, `channel_id`, `question`, `answers` (2-10), `duration_hours`, `multiple`, `layout_type` | `send(poll=discord.Poll(question, timedelta(hours=n), multiple=..., layout_type=...))` | no |
| `get_poll_results` | `server_id`, `channel_id`, `message_id` | fetch message, read `message.poll` | no |
| `forward_message` | `server_id`, `channel_id`, `message_id`, `destination_channel_id` | `message.forward(destination, fail_if_not_exists=True)` | no |
| `pin_message` | `server_id`, `channel_id`, `message_id`, `reason` | `message.pin(reason=...)` | yes |
| `unpin_message` | `server_id`, `channel_id`, `message_id`, `reason` | `message.unpin(reason=...)` | yes |
| `clear_message_reactions` | `server_id`, `channel_id`, `message_id`, `reason` | `message.clear_reactions()` | yes |
| `get_reaction_users` | `server_id`, `channel_id`, `message_id`, `emoji`, `limit` | `async for u in message.reactions[0].users(limit=...)` | no |
| `create_thread_from_message` | `server_id`, `channel_id`, `message_id`, `name`, opt `auto_archive_duration`, `slowmode_delay`, `reason` | `message.create_thread(name=..., auto_archive_duration=..., slowmode_delay=None, reason=None)` | yes |
| `send_typing` | `server_id`, `channel_id` | `Messageable.typing()` context manager | no |

Notes: build `discord.AllowedMentions` from the mention flags.
`send_typing` must enter the `typing()` async context manager and exit it cleanly.
`components` spec shape (document it in the schema description):
`[{"type":"button","custom_id":"...","label":"...","emoji":"...","style":1,"disabled":false,"row":0},
{"type":"select","custom_id":"...","placeholder":"...","options":[{"label":"a","value":"a","description":"...","emoji":"..."}],"min_values":1,"max_values":1,"row":0}]`
Map `style` int → `discord.ButtonStyle`, validate range, reject unknown `type` with a clear
error. File loading: `discord.File(path)` for local paths and
`discord.File(io.BytesIO(urlopen(url).read), filename=...)` for URLs — reject a non-http(s)
scheme with a clear error and never shell out.

## Domain A4 — `channel_advanced`

Files: `schemas/channel_advanced.py` → `CHANNEL_ADVANCED_TOOLS`;
`handlers/channel_advanced.py`.

| Tool | Args | discord.py 2.7.1 | Gate |
|---|---|---|---|
| `clone_channel` | `server_id`, `channel_id`; opt `name`, `reason` | `channel.clone(name=..., reason=...)` | yes |
| `create_announcement_channel` | `server_id`, `name`; opt `category_id`, `topic`, `nsfw`, `slowmode_delay`, `default_auto_archive_duration`, `default_thread_slowmode_delay`, `position`, `reason` | `guild.create_text_channel(name, news=True, ...)` | yes |
| `create_stage_channel` | `server_id`, `name`; opt `category_id`, `bitrate`, `user_limit`, `rtc_region`, `video_quality_mode`, `nsfw`, `position`, `reason` | `guild.create_stage_channel(...)` | yes |
| `follow_channel` | `server_id`, `channel_id`, `webhook_channel_id` | `text_channel.follow(webhook_channel)` | yes |
| `sync_channel_permissions` | `server_id`, `category_id`; opt `reason` | no single call — MCP-side: copy the category's `overwrites` onto each child via `child.set_permissions(...)` | yes |
| `set_voice_channel_status` | `server_id`, `channel_id`, `status`; opt `reason` | `voice_channel.edit(status=..., reason=...)` — travels through `**options` → `abc.py:564` → `http.edit_voice_channel_status` (contradicts the gap doc's §3 N/A claim; see Verified corrections) | yes |

Notes: `sync_channel_permissions` must be documented honestly as a client-side helper that
mirrors the category overwrite onto every child channel, matching what Discord's UI does on
category sync. It must read each child's *current* overwrite and report
`{"channelId","previous","applied"}` per channel in the dry-run details so the operator can
see the diff before executing.

## Domain B1 — `members_roles_advanced`

Files: `schemas/members_roles_advanced.py` → `MEMBERS_ROLES_ADVANCED_TOOLS`;
`handlers/members_roles_advanced.py`.

| Tool | Args | discord.py 2.7.1 | Gate |
|---|---|---|---|
| `change_member_voice_state` | `server_id`, `member_id`; opt `channel_id`, `mute`, `deafen`, `reason` | `member.edit(voice_channel=, mute=, deafen=, reason=)` | yes |
| `move_member_voice` | `server_id`, `member_id`; opt `channel_id` (`None`/empty = disconnect), `reason` | `Member.move_to(channel_or_None, reason=...)` | yes |
| `request_to_speak` | `server_id`, `member_id`, `reason` | `member.request_to_speak()` | yes |
| `get_member_voice_state` | `server_id`, `member_id` | `Member.voice` / `await member.fetch_voice()` | no |
| `edit_member_profile` | `server_id`, `member_id`; opt `nickname`, `avatar_url`, `banner_url`, `bio`, `reason` | `Member.edit(nick=, avatar=bytes|None, banner=bytes|None, bio=)` | yes |
| `create_dm_channel` | `user_id` | `await user.create_dm()` | no |
| `update_bot_profile` | opt `username`, `avatar_url`, `banner_url` | `await client.user.edit(username=, avatar=, banner=)` | yes |
| `set_role_icon` | `server_id`, `role_id`; opt `icon_url`, `reason` (`icon_url` absent or `null` clears the icon) | `Role.edit(display_icon=bytes|None)` | yes |
| `reorder_roles` | `server_id`, `positions` (map roleId→int), `reason` | `guild.edit_role_positions(positions, reason=...)` | yes |
| `get_role_details` | `server_id`; opt `role_id` | `Role.tags`, `Role.is_bot_managed()`, `Role.is_integration()`, `guild.role_member_counts()` | no |

Notes: `Guild.change_voice_state` only changes the **bot's own** voice state (gateway
opcode 4) and takes no member, so member voice must go through
`Member.edit(voice_channel=, mute=, deafen=)`; `move_to(None)` disconnects from voice —
map an empty/absent `channel_id` to `None` and say so in the schema description.
`edit_member_profile` needs `bytes`: load the
URL the same way `send_message_with_files` does. `update_bot_profile` must use
`deps["discord_client"].user`, not `Client.edit` (which does not exist in 2.7.1).

## Domain B2 — `emoji_sticker_soundboard`

Files: `schemas/emoji_sticker_soundboard.py` → `EMOJI_STICKER_SOUNDBOARD_TOOLS`;
`handlers/emoji_sticker_soundboard.py`.

| Tool | Args | discord.py 2.7.1 | Gate |
|---|---|---|---|
| `create_emoji` | `server_id`, `name`, `image_url`; opt `role_ids`, `reason` | `guild.create_custom_emoji(name=, image=bytes, roles=, reason=)` | yes |
| `edit_emoji` | `server_id`, `emoji_id`; opt `name`, `role_ids`, `reason` | `Emoji.edit(name=, roles=, reason=)` | yes |
| `delete_emoji` | `server_id`, `emoji_id`, `reason` | `guild.delete_emoji(emoji, reason=)` | yes, reason required |
| `create_application_emoji` | `name`, `image_url` | `client.create_application_emoji(name=, image=bytes)` | yes |
| `edit_application_emoji` | `emoji_id`; opt `name` | `await emoji.edit(name=...)` | yes |
| `delete_application_emoji` | `emoji_id` | `client.http.delete_application_emoji(emoji_id)` — verify with `dir()` first; if absent use `await emoji.delete()` | yes |
| `list_application_emojis` | — | `client.fetch_application_emojis()` | no |
| `create_sticker` | `server_id`, `name`, `description`, `emoji`, `file_path`, `reason` | `guild.create_sticker(name=, description=, emoji=str, file=discord.File, reason=)` | yes |
| `delete_sticker` | `server_id`, `sticker_id`, `reason` | `guild.delete_sticker(sticker, reason=)` | yes, reason required |
| `list_stickers` | `server_id` | `guild.fetch_stickers()` | no |
| `create_soundboard_sound` | `server_id`, `name`, `sound_url`, opt `volume`, `emoji`, `reason` | `guild.create_soundboard_sound(name=, sound=bytes, volume=1, emoji=None, reason=)` | yes |
| `list_soundboard_sounds` | `server_id` | `guild.fetch_soundboard_sounds()` | no |
| `edit_soundboard_sound` | `server_id`, `sound_id`; opt `name`, `volume`, `emoji`, `reason` | `sound.edit(name=, volume=, emoji=, reason=)` | yes |
| `delete_soundboard_sound` | `server_id`, `sound_id`, `reason` | `sound.delete(reason=)` | yes, reason required |
| `send_soundboard_sound` | `server_id`, `channel_id`, `sound_id` | `VoiceChannel.send_sound(sound)` | no |

Notes: **reuse** `core/emoji.py` for emoji naming/encoding — read it before writing any
emoji string handling, and do not re-implement a second codec. `list_guild_emojis` already
exists, so do **not** add a duplicate emoji-list tool. Verify
`delete_application_emoji` against the installed client before coding it.

## Domain B3 — `webhook_mgmt`

Files: `schemas/webhook_mgmt.py` → `WEBHOOK_MGMT_TOOLS`;
`handlers/webhook_mgmt.py`.

| Tool | Args | discord.py 2.7.1 | Gate |
|---|---|---|---|
| `get_webhook` | `webhook_id`, `webhook_token` | `client.fetch_webhook(id, token=token)` / `gateway.fetch_webhook` | no |
| `edit_webhook` | `webhook_id`, `webhook_token`; opt `name`, `channel_id`, `avatar_url`, `reason` | `Webhook.edit(name=, channel=, avatar=)` | yes |
| `delete_webhook` | `webhook_id`, `webhook_token`, `reason` | `Webhook.delete(reason=)` | yes, reason required |
| `get_webhook_message` | `webhook_id`, `webhook_token`, `message_id` | `Webhook.fetch_message(message_id)` | no |
| `edit_webhook_message` | `webhook_id`, `webhook_token`, `message_id`; opt `content` | `Webhook.edit_message(message_id, content=)` | yes |
| `delete_webhook_message` | `webhook_id`, `webhook_token`, `message_id`, `reason` | `Webhook.delete_message(message_id)` | yes, reason required |

Notes: reuse `gateway.fetch_webhook(webhook_id, token)`; do not re-implement the lookup.
Never echo the webhook **token** back in a payload — mask it (e.g. last 4 chars).

## Domain B4 — `scheduled_stage`

Files: `schemas/scheduled_stage.py` → `SCHEDULED_STAGE_TOOLS`;
`handlers/scheduled_stage.py`.

| Tool | Args | discord.py 2.7.1 | Gate |
|---|---|---|---|
| `create_scheduled_event` | `server_id`, `name`, `entity_type`, `start_time`; opt `description`, `channel_id`, `end_time`, `privacy_level`, `image_url`, `location`, `reason` | `Guild.create_scheduled_event(...)` | yes |
| `get_scheduled_event` | `server_id`, `event_id`, opt `with_user_count` | `guild.fetch_scheduled_event(id, with_user_count=)` | no |
| `list_scheduled_events` | `server_id`; opt `with_user_count` | `guild.fetch_scheduled_events(with_user_count=)` | no |
| `edit_scheduled_event` | `server_id`, `event_id`; opt `name`, `description`, `channel_id`, `start_time`, `end_time`, `privacy_level`, `entity_type`, `image_url`, `location`, `reason` | `ScheduledEvent.edit(...)` | yes |
| `delete_scheduled_event` | `server_id`, `event_id`, `reason` | `ScheduledEvent.delete(reason=)` | yes, reason required |
| `start_scheduled_event` | `server_id`, `event_id`, `reason` | `ScheduledEvent.start(reason=)` | yes |
| `end_scheduled_event` | `server_id`, `event_id`, `reason` | `ScheduledEvent.end(reason=)` | yes |
| `cancel_scheduled_event` | `server_id`, `event_id`, `reason` | `ScheduledEvent.cancel(reason=)` | yes, reason required |
| `list_scheduled_event_users` | `server_id`, `event_id`; opt `limit`, `with_member` | `async for u in event.users(limit=, with_member=)` | no |
| `create_stage_instance` | `server_id`, `channel_id`, `topic`; opt `privacy_level`, `send_start_notification`, `scheduled_event_id`, `reason` | `StageChannel.create_instance(topic=, privacy_level=, send_start_notification=, scheduled_event=, reason=)` | yes |
| `get_stage_instance` | `server_id`, `channel_id` | `StageChannel.fetch_instance()` | no |
| `edit_stage_instance` | `server_id`, `channel_id`; opt `topic`, `privacy_level`, `reason` | `StageInstance.edit(topic=, privacy_level=, reason=)` | yes |
| `delete_stage_instance` | `server_id`, `channel_id`, `reason` | `StageInstance.delete(reason=)` | yes, reason required |

Notes: `discord.EntityType` (`stage_instance`/`voice_channel`/`external`) and
`discord.EventStatus` must be parsed from names with a clear error on an unknown value.
`start_time`/`end_time` arrive as ISO-8601 strings — parse with
`datetime.datetime.fromisoformat`, accepting a trailing `Z`.

## Domain C1 — `templates_widget`

Files: `schemas/templates_widget.py` → `TEMPLATES_WIDGET_TOOLS`;
`handlers/templates_widget.py`.

| Tool | Args | discord.py 2.7.1 | Gate |
|---|---|---|---|
| `list_templates` | `server_id` | `guild.templates()` | no |
| `create_template` | `server_id`, `name`; opt `description`, `reason` | `guild.create_template(name=, description=)` | yes |
| `get_template` | `code` (template code or URL) | `client.fetch_template(code)` | no |
| `sync_template` | `code`; opt `name`, `description` | `template.sync()` (sync before edit if you need to mutate) | yes |
| `edit_template` | `code`; opt `name`, `description` | `template.edit(name=, description=)` | yes |
| `delete_template` | `code` | `template.delete()` | yes, reason optional arg `reason` |
| `get_guild_preview` | `server_id` | `client.fetch_guild_preview(guild_id)` | no |
| `get_widget_settings` | `server_id` | `guild.widget()` → `Widget(name, channel, invite_url, json_url, presence_count)` | no |
| `edit_widget_settings` | `server_id`; opt `enabled`, `channel_id`, `reason` | `guild.edit_widget(enabled=, channel=, reason=)` | yes |

Notes: `fetch_template` accepts either the raw code or a `discord.gg/<code>` URL — strip the
prefix. Templates are **account-level**, not guild-level, so these tools take `code`, not
`server_id`, except the three that need a guild (`list_templates`, `get_guild_preview`,
widget tools).

## Domain C2 — `monetization_appcmds`

Files: `schemas/monetization_appcmds.py` → `MONETIZATION_APPCMDS_TOOLS`;
`handlers/monetization_appcmds.py`.

| Tool | Args | discord.py 2.7.1 | Gate |
|---|---|---|---|
| `list_skus` | — | `client.fetch_skus()` | no |
| `list_entitlements` | opt `limit`, `sku_ids`, `user_id`, `guild_id`, `exclude_ended`, `exclude_deleted` | `async for e in client.entitlements(...)` | no |
| `get_entitlement` | `entitlement_id` | `client.fetch_entitlement(id)` | no |
| `create_entitlement` | `sku_id`, `owner_id`, `owner_type`, `reason` | `client.create_entitlement(sku, owner, EntitlementOwnerType)` | yes |
| `consume_entitlement` | `entitlement_id` | `await entitlement.consume()` | yes |
| `delete_entitlement` | `entitlement_id`, `reason` | `await entitlement.delete()` | yes, reason required |
| `list_app_commands` | opt `guild_id` | `client.tree.fetch_commands(guild=...)` | no |
| `get_app_command` | `command_id`; opt `guild_id` | `client.tree.fetch_command(id, guild=...)` | no |
| `sync_app_commands` | opt `guild_id` | `client.tree.sync(guild=...)` | yes |

Notes: the monetization tools only do anything for a monetized application — when
`fetch_skus` returns an empty list the schema description must say so. `sync_app_commands`
rewrites the application's command tree and can take minutes to propagate; say that in the
description and return `{"synced": n, "guildId": ...}`. Entitlement owner type parses from
`user`/`guild` into `discord.EntitlementOwnerType`.

---

## Total

**96 new tools** across 10 domains, on top of the existing 116 → **212**.

Two tools are not in the original gap list and were added after verification found
them missing or wrongly dismissed: `edit_sticker` (the gap document omitted
`GuildSticker.edit`, which does exist in 2.7.1) and `set_voice_channel_status`
(the gap document claimed `VoiceChannel.edit(status=)` does not exist; it does).

## Live verification

`live_smoke.py` in the repo root drives the new tools against a real Discord server
through the real handler layer. It resolves the guild from the standard
`DEFAULT_GUILD_ID` environment variable, records every artifact it creates, deletes
them all in a `finally` block, and then re-reads server state to prove nothing
survived. Run it with the usual `DISCORD_TOKEN` / `DISCORD_MCP_CONFIRM_SECRET`
environment in place.

Three tools cannot be exercised on an arbitrary server and fail with Discord's own
error rather than a defect:

- `change_member_voice_state` needs a member already connected to voice (error 40032).
- `create_sticker` needs a free sticker slot; Discord caps a guild at 5 (error 30039).
- `create_soundboard_sound` needs Opus or MP3 audio; other containers are rejected.

`list_app_commands` / `get_app_command` / `sync_app_commands` require a client that
has an application command tree. `server.py` runs `discord.ext.commands.Bot`, which
does; a bare `discord.Client` does not, and the tools say so rather than raising
`AttributeError`.

Two unit-test-level environment notes:

- `create_stage_instance`, `get_stage_instance`, `edit_stage_instance` and
  `delete_stage_instance` need a channel of type **stage**.
- The monetization tools only do anything for a monetized application; a
  non-monetized app simply returns an empty SKU list.