#!/usr/bin/env python3
"""Đo KPI server Bên Hiên Nhà (Discord) — chạy định kỳ để so với baseline.

Cách chạy:
    cd ~/Work/FOSS/mcp-discord
    .venv/bin/python docs/ops/measure_bhn_metrics.py

Đọc token từ ~/Work/FOSS/mcp-discord/.env (DISCORD_TOKEN).
Kết quả: docs/ops/bhn-metrics-<YYYY-MM-DD>.json + tóm tắt in ra stdout.

Baseline 2026-10-07 (để so):
    member thường thấy 27 kênh / ẩn 31; mod thấy 34
    MAU (30d) 30/274 = 11%; WAU (7d) 13/274 = 4.7%
    tin nhắn người 30d >=699 (~23/ngày); top-3 chiếm 42%, top-10 chiếm 81%
    activation người mới <=30d: 38% · <=90d: 22%
    kênh public có >=1 tin/30d: 15/41 · forum: 0 thread/90d
    giờ cao điểm: 19-23h = 43% hoạt động
"""

from __future__ import annotations

import asyncio
import collections
import datetime
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import discord  # noqa: E402

GUILD = 1424116735782682778
INTERNAL_CAT = 1425719314736091147
SAMPLE_MEMBER = 1188824485533601903  # member thường
SAMPLE_MOD = 627871116916031510      # 👮Quản trị viên
DAYS = 30
WEEKS = 6


def load_env() -> None:
    path = os.path.join(REPO, ".env")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


async def collect(client: discord.Client) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc)
    w30 = now - datetime.timedelta(days=DAYS)
    w42 = now - datetime.timedelta(days=WEEKS * 7)
    w90 = now - datetime.timedelta(days=90)

    guild = await client.fetch_guild(GUILD)
    await guild.fetch_roles()
    channels = await guild.fetch_channels()
    texts = [c for c in channels if isinstance(c, discord.TextChannel)]
    forums = [c for c in channels if isinstance(c, discord.ForumChannel)]
    cats = {c.id: c.name for c in channels if isinstance(c, discord.CategoryChannel)}

    members = [m async for m in guild.fetch_members(limit=None)]
    humans = [m for m in members if not m.bot]

    out: dict = {
        "as_of": now.isoformat(),
        "members": {"total": len(members), "humans": len(humans), "bots": len(members) - len(humans)},
        "cohort_by_join_month": dict(sorted(collections.Counter(m.joined_at.strftime("%Y-%m") for m in humans).items())),
        "joined_dates": {str(m.id): m.joined_at.date().isoformat() for m in humans},
    }

    per_channel, authors, hours, days = {}, collections.Counter(), collections.Counter(), collections.Counter()
    authors_7d = set()
    for ch in texts:
        if ch.category_id == INTERNAL_CAT:
            continue
        n = 0
        async for msg in ch.history(limit=500, after=w30):
            n += 1
            if msg.author.bot:
                continue
            authors[msg.author.id] += 1
            hours[msg.created_at.astimezone().hour] += 1
            days[msg.created_at.date().isoformat()] += 1
            if msg.created_at >= now - datetime.timedelta(days=7):
                authors_7d.add(msg.author.id)
        per_channel[ch.name] = {"count": n, "cat": cats.get(ch.category_id)}

    long_authors = collections.Counter()
    for ch in texts:
        if ch.category_id == INTERNAL_CAT:
            continue
        async for msg in ch.history(limit=800, after=w42):
            if not msg.author.bot:
                long_authors[msg.author.id] += 1

    out["msg"] = {
        "human_messages_30d": sum(authors.values()),
        "unique_authors_30d": len(authors),
        "unique_authors_7d": len(authors_7d),
        "human_messages_42d": sum(long_authors.values()),
        "unique_authors_42d": len(long_authors),
        "authors_42d": {str(k): v for k, v in long_authors.items()},
        "per_channel_30d": dict(sorted(per_channel.items(), key=lambda kv: -kv[1]["count"])),
        "per_hour_30d": dict(sorted(hours.items())),
        "per_day_30d": dict(sorted(days.items())),
    }

    forums_out = {}
    for f in forums:
        total = recent90 = recent30 = 0
        try:
            async for t in f.archived_threads(limit=200):
                total += 1
                recent90 += t.created_at >= w90
                recent30 += t.created_at >= w30
        except Exception as exc:  # pragma: no cover - API hiccup
            forums_out[f.name] = {"error": str(exc)}
            continue
        forums_out[f.name] = {"threads_total": total, "threads_90d": recent90, "threads_30d": recent30}
    out["forums"] = forums_out

    mod = collections.Counter()
    async for entry in guild.audit_logs(limit=1000, after=w30):
        mod[str(entry.action).split(".")[-1]] += 1
    out["mod_actions_30d"] = dict(mod.most_common())

    try:
        invites = await guild.invites()
        out["invites"] = {i.code: {"uses": i.uses, "max_age": i.max_age, "inviter": str(getattr(i.inviter, "name", None))} for i in invites}
    except Exception as exc:  # pragma: no cover
        out["invites"] = {"error": str(exc)}

    visibility = {}
    for label, uid in (("member", SAMPLE_MEMBER), ("mod", SAMPLE_MOD)):
        member = await guild.fetch_member(uid)
        visible = [c for c in channels if c.type != discord.ChannelType.category and c.permissions_for(member).view_channel]
        visibility[label] = {
            "name": member.display_name,
            "visible": len(visible),
            "hidden": len(channels) - len(visible) - len([c for c in channels if c.type == discord.ChannelType.category]),
            "visible_names": sorted(c.name for c in visible),
        }
    out["visibility"] = visibility

    out["roles"] = {
        r.name: sum(1 for m in humans if any(x.id == r.id for x in m.roles))
        for r in guild.roles
        if not r.managed
    }
    return out


def summarize(data: dict) -> None:
    humans = data["members"]["humans"]
    msg = data["msg"]
    print(f"members: {data['members']['total']} (human {humans})")
    print(f"MAU 30d: {msg['unique_authors_30d']}/{humans} = {100*msg['unique_authors_30d']/humans:.0f}%")
    print(f"WAU  7d: {msg['unique_authors_7d']}/{humans} = {100*msg['unique_authors_7d']/humans:.0f}%")
    print(f"tin nhắn người 30d: {msg['human_messages_30d']} (~{msg['human_messages_30d']/DAYS:.0f}/ngày)")
    top = sorted(msg["authors_42d"].values(), reverse=True)
    tot = sum(top) or 1
    print(f"tập trung 42d: top1 {100*top[0]/tot:.0f}% · top3 {100*sum(top[:3])/tot:.0f}% · top10 {100*sum(top[:10])/tot:.0f}%")
    joined = data["joined_dates"]
    today = datetime.date.today()
    for win in (30, 90):
        new = [u for u, d in joined.items() if (today - datetime.date.fromisoformat(d)).days <= win]
        act = [u for u in new if u in msg["authors_42d"]]
        print(f"activation <= {win}d: {len(act)}/{len(new)} = {100*len(act)/max(len(new),1):.0f}%")
    dead = [k for k, v in msg["per_channel_30d"].items() if v["count"] == 0]
    print(f"kênh public 0 tin/30d: {len(dead)}/{len(msg['per_channel_30d'])}")
    print(f"forum thread 90d: { {k: v.get('threads_90d') for k, v in data['forums'].items()} }")
    print(f"sidebar: {data['visibility']['member']['visible']} kênh cho member · {data['visibility']['mod']['visible']} cho mod")
    print(f"invite đang sống: {len(data.get('invites', {}))}")


async def main() -> None:
    load_env()
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("DISCORD_TOKEN không có trong môi trường hoặc .env")
    intents = discord.Intents.none()
    intents.members = True
    client = discord.Client(intents=intents)
    await client.login(token)
    data = await collect(client)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(out_dir, f"bhn-metrics-{datetime.date.today().isoformat()}.json")
    with open(path, "w") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1, default=str)
    summarize(data)
    print(f"\nwrote {path}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
