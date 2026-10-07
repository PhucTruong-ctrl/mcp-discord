# EMOJI_ROUNDTRIP_ISSUES — brief for whoever works in this repo

Handed over by a sibling agent that spent the day driving the live guild
`1424116735782682778` (Bên Hiên Nhà, bot `Mèo Lùn` whose role sits at the top of the hierarchy).
Every claim below was reproduced against that live guild — the repro lines are real error text, not
code reading. You already have `docs/ops/bhn-metrics-*.json` and `docs/ops/bhn-permission-baseline-*.json`
for the same guild, so you can re-verify live.

P1..P4 are code bugs: each has a repro, the file/line to touch, and an acceptance check.
Nothing here needs product decisions.

---

## P1. `get_guild_onboarding` returns channel NAMES where the write path needs snowflakes

Read returns:

```json
"defaultChannels": ["╭・🧭 Cổng Làng", "🍵・hiên-trước", "📇・sổ-tên", "╭・🎤 Bếp Lửa", "📮・hòm-góp-ý", "⚙️・bảng-phòng"]
```

The write path pipes whatever it receives into `_id_list(...)`, which requires ints:

- `src/discord_mcp/core/serialize.py:198-208` — `_serialize_onboarding`
- `src/discord_mcp/tools/handlers/onboarding.py:361-373` — `raw_default_channels` → `_id_list(raw_default_channels, "onboarding.default_channels")`

A verbatim read → write round trip therefore fails, and a caller holding only the read output cannot fill
the field at all.

Fix: emit ids (`default_channel_ids`) on read. Keep the names as a display-only extra if useful, but the
canonical field must be snowflakes.

Acceptance: read onboarding, send the payload straight back, expect 200 and no field drift.

## P2. Welcome screen drops the custom emoji id (the bug onboarding had before f44b75b)

- `core/serialize.py:4-7` — `_serialize_emoji` returns `emoji.name` only.
- `core/serialize.py:154` — `"emoji": _serialize_emoji(wc.emoji)` per welcome channel.
- `tools/handlers/onboarding.py:13-36` — `_build_welcome_channels` passes that value into
  `discord.WelcomeChannel(emoji=...)`.

Repro: point a welcome-screen channel at a guild emoji, read, write the same payload back. Discord answers:

```
400 Invalid Form Body
In welcome_channels.0.emoji: Invalid emoji id or name
```

Fix: reuse the onboarding codec. Read `{emoji, emojiId, emojiAnimated}`; on write accept
unicode | `"name"` | `"<:name:id>"` | `{id, name, animated}`. `_emoji_token`
(`handlers/onboarding.py:233-240`) already builds the token — move it to a shared module instead of
copying it.

Acceptance: round trip a welcome channel that uses a custom emoji → 200, emoji unchanged.

## P3. Forum tags and `default_reaction_emoji` are write-only

- `core/serialize.py:78` — `_serialize_tag` emits the tag emoji as a bare name, no id.
- No read exposes `available_tags` or `default_reaction_emoji`: `get_channels_structured` returns
  `id, name, type, position, categoryId, topic`; `list_forum_posts` returns tags with a name-only emoji.

So `update_forum_channel.available_tags` can be written but never read back, and a custom tag emoji cannot
survive a round trip.

Fix: same emoji codec for tag emoji; expose `available_tags` (id, name, emoji, moderated) and
`default_reaction_emoji` from `get_channels_structured`, or add a dedicated `get_forum_channel`.

Acceptance: read a forum channel carrying a custom tag emoji, write the payload back → 200, no drift.

## P4. Nothing can list guild emojis

`add_reaction`, `add_multiple_reactions`, `remove_reaction` document "Unicode or custom emoji ID", but no
tool returns those ids. The only workaround today is happenstance: read a message that happens to contain
`<:name:id>` and harvest the id out of the rendered content. That is how the ids in this brief were found,
and it is not a workflow.

Fix: new tool `list_guild_emojis` returning `[{id, name, animated, available, managed, roles,
require_colons}]`, plus optional `include_stickers`. discord.py hands it over via `guild.emojis` /
`guild.stickers`.

Acceptance: returns every emoji in the guild; returned ids match the `<:name:id>` seen in message content.

## Shared contract — build once, then use everywhere

```python
def serialize_emoji(emoji) -> tuple[str | None, str | None, bool]:   # (name, id, animated)
def parse_emoji(guild, value) -> str | None:  # unicode | "name" | "<:name:id>" | {name,id,animated} -> token
```

`parse_emoji` should resolve a bare `"name"` against `guild.emojis` so a name read from anywhere is
usable on write without the caller hand-carrying ids. Onboarding, welcome screen, forum tags and
reactions all route through these two.

---

## Field notes, no code change requested

- `update_guild_onboarding` against a guild with 0 channels writable by `@everyone` returns Discord 350001.
  The error text the tool prints is already precise (27 public channels, 0 writable, needs >= 7 and >= 5).
  Worth one docstring line: the same rule blocks the Discord client, so the remedy is to open 5 channels to
  `@everyone` temporarily, save, then revert.
- Managed roles: `update_role` renames a `managed: true` role fine when it sits below the bot's top role —
  17 bot roles were renamed that way today. The only 403 came from hierarchy (the bot's own top role).
  The opposite assumption is the intuitive one, so it deserves a docs line.
- `get_role_permissions` used to omit `color`/`colorHex` when called without `role_id`. Fixed as of
  commit 284c8c9; the `report_issue` filed about it can be closed.

## Verification expected with the patch

1. `python -m pytest` green, plus new tests for the four round trips (read → write → 200, no drift).
2. Live smoke on guild `1424116735782682778`: onboarding verbatim round trip, welcome screen with a custom
   emoji, forum tag with a custom emoji, `list_guild_emojis` returning that guild's emoji set.
3. Say in the PR/commit that the running MCP process needs a restart before the client sees the fix.

---

## Addendum: P1..P4 verified live, plus one correction to the merge comment

All four were verified against the live guild after the patch:

- P1 — `get_guild_onboarding` now returns `defaultChannelIds` (ids) next to the display names.
- P2 — `get_guild_welcome_screen` now returns `emojiId` / `emojiAnimated` per welcome channel.
- P3 — `list_forum_posts` tag objects now carry `emojiId` / `emojiAnimated`.
- P4 — `list_guild_emojis` returns 84 emojis with `id`, `name`, `animated`, `available`, `managed`,
  `requireColons`, `token`, `url`, `roleIds`, plus `stickerCount` / `stickers`.

Correction, found while doing a real write: **prompt and option ids are not stable across a PUT.**
`handle_update_guild_onboarding` documents the merge as preserving "channel order, prompt ids and animated
emoji detail". The write succeeded, but Discord regenerated every prompt and option id:

```
before: prompt 1494710231895248940, option 1494710231895248943
after : prompt 1557467612491808862, option 1557467612491808865
```

Content survived (titles, descriptions, channel/role ids, emoji id + animated flag all round tripped),
identifiers did not. The PUT does not accept ids back, so this is an API property, not a bug in the handler.
Two follow-ups worth doing:

1. Document it, and drop "prompt ids survive" from that comment — anyone storing option ids as stable keys
   (dashboards, metrics baselines, this repo's `docs/ops/*.json`) gets silently stale references.
2. Since the handler already returns the new ids, a docstring line pointing at that field is enough for
   callers to re-read and re-store.

## Status

- P1 — implemented
- P2 — implemented
- P3 — implemented
- P4 — implemented
- Live-smoke outcome is recorded in the commit message.
- The running MCP process must be restarted before the client sees the fixes.
