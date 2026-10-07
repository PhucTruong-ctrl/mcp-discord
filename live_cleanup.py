"""Unconditional cleanup sweep for anything the live smoke harness may have left.

Deliberately independent of live_smoke.py's in-memory bookkeeping: it re-reads
live server state and removes anything matching the smoke markers, so it works
even if the harness was killed before its finally block ran.

Usage: .venv/bin/python live_cleanup.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import discord  # noqa: E402

MARKER = "smoke"
GUILD_ID = int(os.environ.get("SMOKE_GUILD_ID", "1424116735782682778"))


async def main() -> int:
    client = discord.Client(intents=discord.Intents.default())
    await client.login(os.environ["DISCORD_TOKEN"])
    await client.connect(reconnect=False)
    await client.wait_until_ready()
    guild = await client.fetch_guild(GUILD_ID)
    removed, failed = [], []

    async def sweep(label, items, delete):
        for item in items:
            try:
                await delete(item)
                removed.append(f"{label} {getattr(item, 'name', item)}")
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{label} {getattr(item, 'name', item)}: {exc}")

    # Emoji, stickers, sounds, scheduled events, threads, channels, categories.
    await sweep("emoji", [e for e in await guild.fetch_emojis() if MARKER in (e.name or "")],
                lambda e: guild.delete_emoji(e, reason="live smoke cleanup"))
    await sweep("sticker", [s for s in await guild.fetch_stickers() if MARKER in (s.name or "")],
                lambda s: guild.delete_sticker(s, reason="live smoke cleanup"))
    await sweep("sound", [s for s in await guild.fetch_soundboard_sounds() if MARKER in (s.name or "")],
                lambda s: s.delete(reason="live smoke cleanup"))
    await sweep("event", [e for e in await guild.fetch_scheduled_events() if MARKER in (e.name or "")],
                lambda e: e.delete(reason="live smoke cleanup"))

    for thread in [t for t in getattr(guild, "_threads", {}).values() if MARKER in (t.name or "")]:
        try:
            await thread.delete()
            removed.append(f"thread {thread.name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"thread {thread.name}: {exc}")

    for channel in [c for c in guild.channels if MARKER in (c.name or "")]:
        try:
            await channel.delete(reason="live smoke cleanup")
            removed.append(f"channel #{channel.name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"channel #{channel.name}: {exc}")

    for invite in await guild.invites():
        try:
            await invite.delete(reason="live smoke cleanup")
            removed.append(f"invite {invite.code}")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"invite {invite.code}: {exc}")

    print(f"removed {len(removed)} artifact(s):")
    for item in removed:
        print(f"  - {item}")
    if failed:
        print(f"\nFAILED to remove {len(failed)}:")
        for item in failed:
            print(f"  ! {item}")

    # Verify.
    await asyncio.sleep(2)
    fresh = await client.fetch_guild(GUILD_ID)
    leftovers = {
        "channels": [c.name for c in fresh.channels if MARKER in (c.name or "")],
        "categories": [c.name for c in fresh.categories if MARKER in (c.name or "")],
        "threads": [t.name for t in getattr(fresh, "_threads", {}).values() if MARKER in (t.name or "")],
        "emojis": [e.name for e in await fresh.fetch_emojis() if MARKER in (e.name or "")],
        "stickers": [s.name for s in await fresh.fetch_stickers() if MARKER in (s.name or "")],
        "sounds": [s.name for s in await fresh.fetch_soundboard_sounds() if MARKER in (s.name or "")],
        "events": [e.name for e in await fresh.fetch_scheduled_events() if MARKER in (e.name or "")],
        "invites": [i.code for i in await fresh.invites()],
    }
    leftovers = {k: v for k, v in leftovers.items() if v}
    await client.close()

    if leftovers:
        print("\nLEFTOVERS STILL PRESENT:", leftovers)
        return 1
    print("\nVERIFIED CLEAN: no smoke artifacts remain on the server.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))