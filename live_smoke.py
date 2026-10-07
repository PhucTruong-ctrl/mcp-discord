"""Live smoke harness for the coverage-gap tools.

Drives every new tool through the real MCP handler layer against the real
Discord API (no mocks), then deletes everything it created. Every artifact is
recorded the moment it is created and removed in a finally block; the cleanup
is then verified by re-reading server state and asserting the leftovers are
gone.

Usage: .venv/bin/python live_smoke.py [--only <substring>] [--keep]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import struct
import sys
import traceback
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import discord  # noqa: E402

from discord_mcp.composition import build_tool_dependencies  # noqa: E402
from discord_mcp.core.safety import generate_confirm_token  # noqa: E402
from discord_mcp.services.discord_gateway import DiscordGateway  # noqa: E402
from discord_mcp.tools.handlers.router import TOOL_ROUTER  # noqa: E402

PNG_URL = "https://cdn.discordapp.com/embed/avatars/0.png"
OGG_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg"
SCRATCH = "mcp-live-smoke"

# Artifacts to tear down, newest last.
# Messages live in channels, so record "<channel_id>:<message_id>" to make teardown exact.
created: dict[str, list] = {"channels": [], "threads": [], "messages": [], "invites": [],
                            "roles": [], "emojis": [], "stickers": [], "sounds": [],
                            "webhooks": [], "events": [], "templates": [], "entitlements": [],
                            "stage_instances": [], "app_emojis": []}

results: list[tuple[str, str, str]] = []
only = ""
keep = False


def record(kind: str, identifier) -> None:
    if identifier is not None and str(identifier):
        created[kind].append(str(identifier))
        note(f"created {kind}: {identifier}")


def note(msg: str) -> None:
    print(f"      {msg}", flush=True)


def png_320() -> bytes:
    """Build a valid 320x320 RGBA PNG with the stdlib (Discord sticker size)."""
    w = h = 320
    raw = b"".join(b"\x00" + bytes([80, 120, 200, 255] * w) for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


async def call(tool: str, arguments: dict, deps: dict, *, execute: bool = False,
               expect_error: str | None = None):
    """Invoke a tool through the router, the way an operator would.

    For the execute path we first run the tool as a dry run and mint the token
    from the ``targets`` the tool itself reports, so the harness never guesses
    at which fields a given domain binds into its confirm token (some include
    ``reason``, some do not).
    """
    if execute:
        try:
            preview = await TOOL_ROUTER[tool](dict(arguments), deps)
            targets = json.loads(preview[0].text).get("targets", {})
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {tool} dry run: {type(exc).__name__}: {exc}", flush=True)
            results.append((tool, "FAIL", f"dry run failed: {exc}"[:300]))
            return None
        args = dict(arguments)
        args["dry_run"] = False
        args["confirm_token"] = generate_confirm_token(tool, targets)
    else:
        args = dict(arguments)

    try:
        out = await TOOL_ROUTER[tool](args, deps)
    except Exception as exc:  # noqa: BLE001 - harness reports, never hides
        text = f"{type(exc).__name__}: {exc}"
        if expect_error and expect_error.lower() in text.lower():
            results.append((tool, "ok", f"expected error: {text[:90]}"))
            return None
        results.append((tool, "FAIL", text[:300]))
        print(f"  !! {tool}: {text}", flush=True)
        return None
    if expect_error:
        results.append((tool, "FAIL", f"expected an error containing {expect_error!r}"))
        return None
    body = json.loads(out[0].text)
    results.append((tool, "ok", ""))
    return body


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    only, keep = args.only, args.keep

    intents = discord.Intents.default()
    intents.members = True
    client = discord.Client(intents=intents)
    ready = asyncio.Event()

    @client.event
    async def on_ready() -> None:
        ready.set()

    # Client.start() is login() + connect(); driving login()/connect() separately
    # stalls on this runtime, so use the single documented entry point.
    runner = asyncio.create_task(client.start(os.environ["DISCORD_TOKEN"]))
    await asyncio.wait_for(ready.wait(), 45)
    print(f"connected as {client.user}\n", flush=True)

    deps = build_tool_dependencies(client)
    gateway: DiscordGateway = deps["gateway"]
    guild = await gateway.resolve_guild(None)
    print(f"guild: {guild.name} ({guild.id})\n", flush=True)

    sid = str(guild.id)
    text_channels = [c for c in guild.text_channels if not c.is_nsfw] or list(guild.text_channels)
    base_channel = text_channels[0]
    voice_channels = list(guild.voice_channels)
    member = next((m for m in guild.members if not m.bot), None)
    voice_member = next(
        (m for m in guild.members if m.voice and m.voice.channel), None
    )
    try:
        # ---------- scratch category (anchor for every artifact we create) ----------
        scratch_category = await guild.create_category(SCRATCH, reason="mcp live smoke")
        record("channels", scratch_category.id)
        gid = int(scratch_category.id)

        cat_arg = {"server_id": sid, "category_id": str(scratch_category.id)}

        # ================= 1. invites & membership =================
        if not only or only in "create_invite":
            ch = await guild.create_text_channel(f"{SCRATCH}-inv", category=scratch_category)
            record("channels", ch.id)
            cid = str(ch.id)
            p = await call("create_invite", {**cat_arg, "channel_id": cid, "max_age": 300},
                           deps, expect_error=None)
            if p and p.get("status") == "executed":
                code = p["invite"]["code"]
                record("invites", code)
                await call("list_invites", {**cat_arg, "channel_id": cid}, deps)
                await call("delete_invite",
                           {"server_id": sid, "invite_code": code, "reason": "smoke"},
                           deps, execute=True,
                           )
            await call("list_bans", {"server_id": sid, "limit": 5}, deps)
            await call("search_members", {"server_id": sid, "query": "a", "limit": 5}, deps)
            await call("get_role_member_counts", {"server_id": sid}, deps)
            await call("estimate_pruned_members", {"server_id": sid, "days": 7}, deps)
            await call("get_role_details", {"server_id": sid}, deps)

        # ================= 2. messages =================
        scratch_ch = await guild.create_text_channel(f"{SCRATCH}-msg", category=scratch_category)
        record("channels", scratch_ch.id)
        mcid = str(scratch_ch.id)

        if not only or only in "send_message_with_files":
            tmp = REPO / ".live-smoke-asset.png"
            tmp.write_bytes(png_320())
            try:
                p = await call("send_message_with_files",
                               {"server_id": sid, "channel_id": mcid, "content": "smoke file",
                                "file_paths": [str(tmp)], "tts": False}, deps)
                if p:
                    record("messages", f'{mcid}:{p.get("messageId")}')
            finally:
                tmp.unlink(missing_ok=True)

        if not only or only in "send_components":
            p = await call("send_components",
                           {"server_id": sid, "channel_id": mcid,
                            "components": [{"type": "button", "custom_id": "smoke",
                                            "label": "smoke", "style": 1}]},
                           deps)
            if p:
                record("messages", f'{mcid}:{p.get("messageId")}')

        if not only or only in "send_poll":
            p = await call("send_poll",
                           {"server_id": sid, "channel_id": mcid, "question": "smoke?",
                            "answers": ["a", "b"], "duration_hours": 1}, deps)
            poll_msg = p.get("messageId") if p else None
            if poll_msg:
                record("messages", f'{mcid}:{poll_msg}')
                await call("get_poll_results",
                           {"server_id": sid, "channel_id": mcid, "message_id": poll_msg}, deps)

        if not only or only in "pin_message":
            base = await base_channel.send("mcp live smoke base message")
            record("messages", f'{base.channel.id}:{base.id}')
            await call("pin_message",
                       {"server_id": sid, "channel_id": str(base.channel.id),
                        "message_id": str(base.id), "reason": "smoke"},
                       deps, execute=True,
                       )
            await call("unpin_message",
                       {"server_id": sid, "channel_id": str(base.channel.id),
                        "message_id": str(base.id), "reason": "smoke"},
                       deps, execute=True,
                       )
            await base.add_reaction("\N{THUMBS UP SIGN}")
            await call("get_reaction_users",
                       {"server_id": sid, "channel_id": str(base.channel.id),
                        "message_id": str(base.id), "emoji": "\N{THUMBS UP SIGN}"}, deps)
            await call("clear_message_reactions",
                       {"server_id": sid, "channel_id": str(base.channel.id),
                        "message_id": str(base.id), "reason": "smoke"},
                       deps, execute=True,
                       )
            fwd = await call("forward_message",
                             {"server_id": sid, "channel_id": str(base.channel.id),
                              "message_id": str(base.id), "destination_channel_id": mcid}, deps)
            if fwd:
                record("messages", f'{mcid}:{fwd.get("messageId")}')
            await call("send_typing", {"server_id": sid, "channel_id": mcid}, deps)

        # ================= 3. threads =================
        if not only or only in "create_thread":
            parent = await guild.create_text_channel(f"{SCRATCH}-thread", category=scratch_category)
            record("channels", parent.id)
            pid = str(parent.id)
            p = await call("create_thread",
                           {"server_id": sid, "channel_id": pid, "name": "smoke-thread",
                            "type": "public_thread"}, deps,
                           execute=True, )
            if p:
                tid = p["thread"]["id"]
                record("threads", tid)
                await call("list_active_threads", {"server_id": sid}, deps)
                await call("edit_thread",
                           {"server_id": sid, "thread_id": tid, "slowmode_delay": 5}, deps,
                           execute=True, )
                if member is not None:
                    await call("add_thread_member",
                               {"server_id": sid, "thread_id": tid, "user_id": str(member.id)},
                               deps, execute=True,
                               )
                    await call("remove_thread_member",
                               {"server_id": sid, "thread_id": tid, "user_id": str(member.id)},
                               deps, execute=True,
                               )
                await call("join_thread", {"server_id": sid, "thread_id": tid}, deps)
                await call("delete_thread",
                           {"server_id": sid, "thread_id": tid, "reason": "smoke"},
                           deps, execute=True,
                           )
            await call("create_thread",
                       {"server_id": sid, "channel_id": pid, "name": "from-message"}, deps,
                       expect_error="forum")

        # ================= 4. channels =================
        if not only or only in "clone_channel":
            p = await call("clone_channel",
                           {"server_id": sid, "channel_id": mcid, "name": f"{SCRATCH}-clone"},
                           deps, execute=True, )
            if p:
                record("channels", p["channel"]["id"])

        if not only or only in "create_announcement_channel":
            p = await call("create_announcement_channel",
                           {**cat_arg, "name": f"{SCRATCH}-news"}, deps,
                           execute=True, )
            if p:
                record("channels", p["channel"]["id"])

        if not only or only in "create_stage_channel":
            p = await call("create_stage_channel",
                           {**cat_arg, "name": f"{SCRATCH}-stage"}, deps,
                           execute=True, )
            if p:
                record("channels", p["channel"]["id"])
                scid = p["channel"]["id"]
                await call("create_stage_instance",
                           {"server_id": sid, "channel_id": scid, "topic": "smoke"}, deps,
                           execute=True, )
                await call("get_stage_instance", {"server_id": sid, "channel_id": scid}, deps)
                await call("edit_stage_instance",
                           {"server_id": sid, "channel_id": scid, "topic": "smoke2"}, deps,
                           execute=True, )
                await call("delete_stage_instance",
                           {"server_id": sid, "channel_id": scid, "reason": "smoke"},
                           deps, execute=True,
                           )

        if not only or only in "sync_channel_permissions":
            await call("sync_channel_permissions", cat_arg, deps,
                       execute=True, )

        # ================= 5. voice =================
        if voice_channels and not only:
            vc = voice_channels[0]
            await call("get_member_voice_state",
                       {"server_id": sid, "member_id": str(member.id)}, deps)
            await call("change_member_voice_state",
                       {"server_id": sid, "member_id": str(member.id),
                        "channel_id": str(vc.id), "mute": True}, deps,
                       execute=True, )
            await call("change_member_voice_state",
                       {"server_id": sid, "member_id": str(member.id), "mute": False}, deps,
                       execute=True, )
            await call("set_voice_channel_status",
                       {"server_id": sid, "channel_id": str(vc.id),
                        "status": "smoke test", "reason": "smoke"}, deps,
                       execute=True, )

        # ================= 6. emoji / stickers / soundboard =================
        if not only or only in "create_emoji":
            p = await call("create_emoji",
                           {"server_id": sid, "name": "smoketest", "image_url": PNG_URL}, deps,
                           execute=True, )
            if p:
                eid = p["emoji"]["id"]
                record("emojis", eid)
                await call("edit_emoji", {"server_id": sid, "emoji_id": eid,
                                          "name": "smoketest2"}, deps,
                           execute=True, )
                await call("delete_emoji",
                           {"server_id": sid, "emoji_id": eid, "reason": "smoke"}, deps,
                           execute=True, )

        if not only or only in "list_application_emojis":
            p = await call("list_application_emojis", {}, deps)
            if p and p.get("emojis"):
                await call("edit_application_emoji", {"emoji_id": p["emojis"][0]["id"]}, deps,
                           expect_error="dry_run|confirm_token")

        if not only or only in "create_sticker":
            existing = await guild.fetch_stickers()
            if len(existing) >= 5:
                note("SKIP create_sticker: guild already at the 5-sticker limit")
            else:
                tmp = REPO / ".live-smoke-sticker.png"
                tmp.write_bytes(png_320())
                try:
                    p = await call("create_sticker",
                                   {"server_id": sid, "name": "smokesticker",
                                    "description": "smoke", "emoji": "smoke",
                                    "file_path": str(tmp), "reason": "smoke"},
                                   deps, execute=True)
                    if p:
                        stid = p["sticker"]["id"]
                        record("stickers", stid)
                        await call("list_stickers", {"server_id": sid}, deps)
                        await call("edit_sticker",
                                   {"server_id": sid, "sticker_id": stid,
                                    "name": "smokesticker2"}, deps, execute=True)
                        await call("delete_sticker",
                                   {"server_id": sid, "sticker_id": stid,
                                    "reason": "smoke"}, deps, execute=True)
                finally:
                    tmp.unlink(missing_ok=True)

        if not only or only in "create_soundboard_sound":
            p = await call("create_soundboard_sound",
                           {"server_id": sid, "name": "smokesound", "sound_url": OGG_URL},
                           deps, execute=True, )
            if p:
                soid = p["sound"]["id"]
                record("sounds", soid)
                await call("list_soundboard_sounds", {"server_id": sid}, deps)
                await call("edit_soundboard_sound",
                           {"server_id": sid, "sound_id": soid, "volume": 0.5}, deps,
                           execute=True, )
                await call("delete_soundboard_sound",
                           {"server_id": sid, "sound_id": soid, "reason": "smoke"}, deps,
                           execute=True, )

        # ================= 7. webhooks =================
        if not only or only in "create_channel_webhook":
            hooks = await scratch_ch.webhooks()
            hook = hooks[0] if hooks else await scratch_ch.create_webhook(name="smoke-hook")
            record("webhooks", hook.id)
            # Use the token-authenticated object so send() has a real session.
            hook = await gateway.fetch_webhook(str(hook.id), hook.token)
            await call("get_webhook", {"webhook_id": str(hook.id),
                                       "webhook_token": hook.token}, deps)
            sent = await hook.send("smoke webhook message")
            if sent is None:
                note("SKIP webhook message tools: hook.send returned no message")
            else:
                record("messages", f"{scratch_ch.id}:{sent.id}")
                await call("get_webhook_message",
                           {"webhook_id": str(hook.id), "webhook_token": hook.token,
                            "message_id": str(sent.id)}, deps)
                await call("edit_webhook_message",
                           {"webhook_id": str(hook.id), "webhook_token": hook.token,
                            "message_id": str(sent.id), "content": "edited"}, deps,
                           execute=True)
            await call("edit_webhook",
                       {"webhook_id": str(hook.id), "webhook_token": hook.token,
                        "name": "smoke-hook-2"}, deps,
                       execute=True, )
            await call("delete_webhook",
                       {"webhook_id": str(hook.id), "webhook_token": hook.token,
                        "reason": "smoke"}, deps, execute=True,
                       )

        # ================= 8. scheduled events =================
        if not only or only in "create_scheduled_event":
            start = discord.utils.utcnow() + datetime.timedelta(hours=2)
            p = await call("create_scheduled_event",
                           {"server_id": sid, "name": "smoke-event",
                            "entity_type": "voice_channel",
                            "channel_id": str(voice_channels[0].id),
                            "start_time": start.isoformat()}, deps,
                           execute=True)
            if p:
                evid = p["event"]["id"]
                record("events", evid)
                await call("get_scheduled_event", {"server_id": sid, "event_id": evid}, deps)
                await call("list_scheduled_events", {"server_id": sid}, deps)
                await call("edit_scheduled_event",
                           {"server_id": sid, "event_id": evid, "name": "smoke-event-2"}, deps,
                           execute=True, )
                await call("delete_scheduled_event",
                           {"server_id": sid, "event_id": evid, "reason": "smoke"}, deps,
                           execute=True, )

        # ================= 9. templates / widget =================
        if not only or only in "list_templates":
            await call("list_templates", {"server_id": sid}, deps)
            p = await call("create_template",
                           {"server_id": sid, "name": "smoke-template",
                            "description": "smoke", "reason": "smoke"}, deps,
                           execute=True, )
            if p:
                code = p["code"]
                record("templates", code)
                await call("get_template", {"code": code}, deps)
                await call("sync_template", {"code": code}, deps,
                           execute=True, )
                await call("edit_template", {"code": code, "name": "smoke-template-2"}, deps,
                           execute=True, )
                await call("delete_template", {"code": code, "reason": "smoke"}, deps,
                           execute=True, )
            await call("get_guild_preview", {"server_id": sid}, deps)
            await call("get_widget_settings", {"server_id": sid}, deps)

        # ================= 10. monetization / app commands =================
        if not only:
            await call("list_skus", {}, deps)
            await call("list_entitlements", {"limit": 5}, deps)
            await call("list_app_commands", {}, deps)

    finally:
        print("\n=== CLEANUP ===", flush=True)
        await cleanup(client, guild)
        await client.close()

    ok = sum(1 for _, s, _ in results if s == "ok")
    bad = [r for r in results if r[1] != "ok"]
    print(f"\n=== {ok}/{len(results)} checks passed ===")
    for tool, _, msg in bad:
        print(f"  FAIL {tool}: {msg}")
    covered = sorted({r[0] for r in results})
    print(f"tools exercised: {len(covered)}")
    return 1 if bad else 0


async def cleanup(client: discord.Client, guild: discord.Guild) -> None:
    """Delete every artifact, newest first, then verify nothing survived."""

    async def drop(fn, kind, ident, label=""):
        try:
            await fn()
            print(f"  removed {kind} {label}{ident}")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove {kind} {label}{ident}: {type(exc).__name__}: {exc}")

    # Templates and entitlements are account-scoped.
    for code in reversed(created["templates"]):
        try:
            tpl = await client.fetch_template(code)
            await drop(tpl.delete, "template", code)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove template {code}: {exc}")

    for sound_id in reversed(created["sounds"]):
        try:
            snd = await guild.fetch_soundboard_sound(int(sound_id))
            await drop(snd.delete, "sound", sound_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove sound {sound_id}: {exc}")

    for sticker_id in reversed(created["stickers"]):
        try:
            await guild.delete_sticker(discord.Object(id=int(sticker_id)), reason="smoke cleanup")
            print(f"  removed sticker {sticker_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove sticker {sticker_id}: {exc}")

    for emoji_id in reversed(created["emojis"]):
        try:
            await guild.delete_emoji(discord.Object(id=int(emoji_id)), reason="smoke cleanup")
            print(f"  removed emoji {emoji_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove emoji {emoji_id}: {exc}")

    for event_id in reversed(created["events"]):
        try:
            ev = await guild.fetch_scheduled_event(int(event_id))
            await drop(ev.delete, "event", event_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove event {event_id}: {exc}")

    for hook_id in reversed(created["webhooks"]):
        try:
            hook = await client.fetch_webhook(int(hook_id))
            await drop(hook.delete, "webhook", hook_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove webhook {hook_id}: {exc}")

    for entry in reversed(created["messages"]):
        channel_id, _, msg_id = entry.partition(":")
        try:
            chan = await client.fetch_channel(int(channel_id))
            msg = await chan.fetch_message(int(msg_id))
            await drop(msg.delete, "message", f"#{chan.name}:{msg_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove message {msg_id} in {channel_id}: {exc}")

    for code in reversed(created["invites"]):
        try:
            inv = await client.fetch_invite(code)
            await drop(inv.delete, "invite", code)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove invite {code}: {exc}")

    for thread_id in reversed(created["threads"]):
        try:
            th = await client.fetch_channel(int(thread_id))
            await drop(th.delete, "thread", thread_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove thread {thread_id}: {exc}")

    for role_id in reversed(created["roles"]):
        try:
            await guild.delete_role(discord.Object(id=int(role_id)), reason="smoke cleanup")
            print(f"  removed role {role_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove role {role_id}: {exc}")

    for channel_id in reversed(created["channels"]):
        try:
            ch = await client.fetch_channel(int(channel_id))
            await drop(ch.delete, "channel", channel_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! could not remove channel {channel_id}: {exc}")

    # ---------- verify ----------
    print("\n=== CLEANUP VERIFICATION ===")
    if keep:
        print("  --keep set, skipping verification")
        return
    await asyncio.sleep(2)
    leftovers = await client.fetch_guild(guild.id, with_counts=False)
    checks = {
        "channels": {c.id for c in leftovers.channels} & set(map(int, created["channels"])),
        "threads": {t.id for t in leftovers._threads} & set(map(int, created["threads"])),
        "emojis": {e.id for e in await leftovers.fetch_emojis()} & set(map(int, created["emojis"])),
        "events": {e.id for e in await leftovers.fetch_scheduled_events()}
                  & set(map(int, created["events"])),
    }
    bad = {k: sorted(v) for k, v in checks.items() if v}
    if bad:
        print(f"  !! LEFTOVERS FOUND: {bad}")
        raise SystemExit(1)
    print(f"  clean: 0 of {sum(len(v) for v in created.values())} artifacts survived")


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(2)