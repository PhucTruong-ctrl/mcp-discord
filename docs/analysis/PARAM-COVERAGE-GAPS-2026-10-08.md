# Parameter-level coverage gaps — 2026-10-08

Status: **all items below implemented** (see "Implemented" column).

`docs/analysis/DISCORDPY_COVERAGE_GAPS.md` audited which discord.py **methods** the registry
exposed. It never compared each tool's **keyword arguments** against the library signature, so a
tool could be "covered" while being unable to do the thing it exists for. `update_role` was the
first proof: discord.py's `Role.edit` takes `position`, the tool did not, and moving a single role
was only reachable through the batch `reorder_roles` dry-run tool.

Method: introspected discord.py 2.7.1 in `.venv` (`inspect.signature`) against the composed tool
registry, plus an AST sweep that flags handler code reading `arguments.get(...)`/`arguments[...]`
keys the tool's JSON schema never declares (that class of bug makes an existing feature
unreachable from any MCP client).

## A. Missing parameters

| Tool | Missing | Evidence | Implemented |
|---|---|---|---|
| `execute_channel_webhook` | embed, attachments, components, `wait`, `avatar_url`, `tts` | `Webhook.send(self, content, *, username, avatar_url, tts, ephemeral, file, files, embed, embeds, allowed_mentions, view, thread, thread_name, wait, suppress_embeds, silent, applied_tags, poll)` vs schema `{webhook_id, token, content, username}` | yes |
| `create_text_channel` | `position`, `nsfw`, `slowmode_delay`, `default_auto_archive_duration`, `default_thread_slowmode_delay` | `Guild.create_text_channel` signature; `update_text_channel` already exposed all of them — create/update were asymmetric | yes |
| `read_messages` | `before` | handler already read `arguments.get("before")` (`handlers/messages.py`), schema declared only `channel_id`/`limit` → history could never be paged | yes |
| `create_voice_channel` | `reason` | handler read it, schema dropped it → audit-log reason silently lost | yes |
| `create_forum_channel` | `reason`, `default_sort_order` (read but undeclared), `position`, `default_layout`, `default_thread_slowmode_delay` (absent) | `Guild.create_forum` signature | yes |
| `update_forum_channel` | `default_layout`, `default_sort_order`, `default_thread_slowmode_delay` | `ForumChannel.edit(**options)` | yes |
| `get_audit_log` | `before`, `after`, `oldest_first` | `Guild.audit_logs(*, limit, before, after, oldest_first, user, action)`; `fetch_audit_entries` only carried limit+action | yes |
| `get_member_moderation_history` | server-side `user` filter | handler fetched one page and filtered client-side (`handlers/audit_analytics.py`), so a member's history was silently truncated | yes |
| `list_members` | `after` | `Guild.fetch_members(*, limit, after)` — guilds over 1000 members were unreachable | yes |
| `list_bans` | `before`, `after` | `Guild.bans(*, limit, before, after)` | yes |
| `prune_inactive_members` | `roles`, `compute_prune_count` | `Guild.prune_members(*, days, compute_prune_count, roles, reason)`; gateway hardcoded `compute_prune_count=True` | yes |

## B. Missing tool

| Tool | Problem | Evidence | Implemented |
|---|---|---|---|
| `delete_auto_moderation_rule` | absent from the registry — a rule could be created and edited but never deleted, so `automod_apply_ruleset` could not make a guild match a ruleset | `AutoModRule.delete(self, *, reason=MISSING)` exists at `.venv/lib/python3.12/site-packages/discord/automod.py:540` | yes |

## C. Destructive tools bypassing the repo's own safety policy

`docs/product/safety/destructive-actions-policy.md` requires dry-run-first + `confirm_token` +
mandatory audit `reason` on destructive actions. `tests/test_confirm_token_enforcement_matrix.py`
only asserted 14 handlers, and none of these were among them — that gap is why they survived.

Gated in this pass: `moderate_message`, `delete_channel`, `delete_role`, `remove_role`,
`remove_reaction` (also gained the `reason` it never had), `remove_channel_permission_overwrite`,
`leave_thread`, `unban_member`, `remove_member_timeout`, `close_incident`,
`incident_set_channel_state`, `create_auto_moderation_rule`, `update_auto_moderation_rule`,
`delete_auto_moderation_rule`.

Every gate now binds `reason` (and the ids it acts on) into the `targets` dict that the confirm
token signs, so a dry-run token issued for one reason cannot be replayed under another.

## D. Checked and NOT gaps

- `display_icon` on a role → `set_role_icon` owns it.
- `create_role` position → `reorder_roles` / `update_role` own position moves.
- `send_message` having only `content` → deliberate; `send_embed_message`,
  `send_message_with_files`, `send_components`, `send_poll` carry the rich surface.
- `create_invite` (`max_age`, `max_uses`, `temporary`, `unique`, `reason`), `create_emoji`/`edit_emoji`
  `reason`, and `dry_run`/`confirm_token` on the emoji/sticker/soundboard/webhook/template/thread/
  category/invite deletes — all already present in the registry.
- AutoMod trigger/action payloads, onboarding and welcome-screen dataclass fields — complete.

## E. How to keep this from recurring

The AST sweep that produced the `read_messages`/`create_voice_channel`/`create_forum_channel`
findings is worth keeping as a check: compare handler argument keys against each tool's schema
properties and fail when a handler reads a key the schema does not declare. That class of defect is
invisible in review — the code works, the tool just cannot reach it.
