import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("DISCORD_TOKEN", "test-token")

import discord_mcp.server as server  # noqa: E402
from discord_mcp.core.resolve import (  # noqa: E402
    _collect_forum_threads,
    _normalize_name,
    _resolve_guild,
    _try_int,
)
from discord_mcp.core.runtime import set_discord_client  # noqa: E402
from discord_mcp.core.serialize import _serialize_message  # noqa: E402
from tests.fakes.discord_fixtures import (  # noqa: E402
    FakeChannel,
    FakeDiscordClient,
    FakeForumTag,
    FakeGuild,
    FakeMessage,
    FakeReaction,
    FakeTagEmoji,
    FakeThread,
    dt,
)


class FakeForumChannel:
    def __init__(self, id, name, guild, threads):
        self.id = id
        self.name = name
        self.guild = guild
        self.threads = threads
        self.available_tags = []


class ToolContractsBaselineTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_channels_contract_plain_text(self):
        guild = FakeGuild(
            id=100,
            name="Acme",
            channels=[
                FakeChannel(id=10, name="general", type="text"),
                FakeChannel(id=11, name="support", type="forum"),
            ],
            text_channels=[],
            member_count=3,
            created_at=dt("2024-01-01T00:00:00"),
        )
        fake_client = FakeDiscordClient(guilds=[guild])

        with patch("discord_mcp.server.discord_client", fake_client):
            result = await server.call_tool("get_channels", {"server_id": "100"})

        self.assertEqual(
            result[0].text,
            "Channels in Acme:\n#general (ID: 10) - text\n#support (ID: 11) - forum",
        )

    async def test_read_messages_contract_plain_text_with_reactions(self):
        messages = [
            FakeMessage(
                id=501,
                author="alice#0001",
                content="Hello",
                created_at=dt("2025-01-01T10:00:00"),
                reactions=[FakeReaction(emoji="🔥", count=2)],
            ),
            FakeMessage(
                id=502,
                author="bob#0002",
                content="World",
                created_at=dt("2025-01-01T10:01:00"),
                reactions=[],
            ),
        ]
        text_channel = FakeChannel(
            id=200, name="general", type="text", _messages=messages
        )
        guild = FakeGuild(
            id=100,
            name="Acme",
            channels=[text_channel],
            text_channels=[text_channel],
            member_count=3,
            created_at=dt("2024-01-01T00:00:00"),
        )
        text_channel.guild = guild
        fake_client = FakeDiscordClient(
            guilds=[guild], channels_by_id={200: text_channel}
        )

        with (
            patch("discord_mcp.server.discord_client", fake_client),
            patch("discord_mcp.server.discord.TextChannel", FakeChannel),
            patch(
                "discord_mcp.server.discord.Thread", type("FakeDiscordThread", (), {})
            ),
        ):
            result = await server.call_tool(
                "read_messages", {"server_id": "100", "channel_id": "200", "limit": 10}
            )

        self.assertEqual(
            result[0].text,
            "Retrieved 2 messages:\n\n"
            "alice#0001 (2025-01-01T10:00:00+00:00): Hello\n"
            "Reactions: 🔥(2)\n"
            "bob#0002 (2025-01-01T10:01:00+00:00): World\n"
            "Reactions: No reactions",
        )

    async def test_list_threads_contract_json_text(self):
        guild = FakeGuild(
            id=100,
            name="Acme",
            channels=[],
            text_channels=[],
            member_count=3,
            created_at=dt("2024-01-01T00:00:00"),
        )
        threads = [
            FakeThread(
                id=901,
                name="Billing issue",
                created_at=dt("2025-02-10T08:00:00"),
                owner_id=12,
                archived=False,
                locked=False,
                message_count=6,
                member_count=2,
                total_message_sent=7,
                slowmode_delay=0,
                applied_tags=[
                    FakeForumTag(id=1, name="help", emoji=FakeTagEmoji(name="🏷️"))
                ],
                last_message_id=930,
            )
        ]
        forum = FakeForumChannel(id=300, name="support", guild=guild, threads=threads)
        guild.channels = [forum]
        fake_client = FakeDiscordClient(guilds=[guild])

        with (
            patch("discord_mcp.server.discord_client", fake_client),
            patch("discord_mcp.server.discord.ForumChannel", FakeForumChannel),
        ):
            result = await server.call_tool(
                "list_threads",
                {"server_id": "100", "channel": "300", "include_archived": False},
            )

        expected = {
            "forumChannel": "support",
            "server": "Acme",
            "includeArchived": False,
            "totalThreads": 1,
            "threads": [
                {
                    "threadId": "901",
                    "threadName": "Billing issue",
                    "createdAt": "2025-02-10T08:00:00+00:00",
                    "ownerId": "12",
                    "archived": False,
                    "locked": False,
                    "messageCount": 6,
                    "memberCount": 2,
                    "totalMessageSent": 7,
                    "rateLimitPerUser": 0,
                    "tags": [
                        {"id": "1", "name": "help", "emoji": "🏷️", "moderated": False}
                    ],
                    "lastMessageId": "930",
                }
            ],
        }
        self.assertEqual(
            result[0].text, json.dumps(expected, ensure_ascii=False, indent=2)
        )

    async def test_search_threads_contract_json_text(self):
        guild = FakeGuild(
            id=100,
            name="Acme",
            channels=[],
            text_channels=[],
            member_count=3,
            created_at=dt("2024-01-01T00:00:00"),
        )
        threads = [
            FakeThread(
                id=1001,
                name="Login problem",
                created_at=dt("2025-02-11T08:00:00"),
                owner_id=13,
                archived=False,
                locked=False,
                message_count=3,
                member_count=2,
                total_message_sent=3,
                slowmode_delay=5,
                applied_tags=[],
                last_message_id=1002,
            ),
            FakeThread(
                id=1003,
                name="Payment issue",
                created_at=dt("2025-02-12T08:00:00"),
                owner_id=14,
                archived=True,
                locked=False,
                message_count=4,
                member_count=2,
                total_message_sent=5,
                slowmode_delay=0,
                applied_tags=[],
                last_message_id=None,
            ),
        ]
        forum = FakeForumChannel(id=300, name="support", guild=guild, threads=threads)
        guild.channels = [forum]
        fake_client = FakeDiscordClient(guilds=[guild])

        with (
            patch("discord_mcp.server.discord_client", fake_client),
            patch("discord_mcp.server.discord.ForumChannel", FakeForumChannel),
        ):
            result = await server.call_tool(
                "search_threads",
                {
                    "server_id": "100",
                    "channel": "300",
                    "query": "issue",
                    "include_archived": False,
                },
            )

        expected = {
            "forumChannel": "support",
            "server": "Acme",
            "query": "issue",
            "exactMatch": False,
            "totalFound": 1,
            "totalMatched": 1,
            "includeArchived": False,
            "threads": [
                {
                    "threadId": "1003",
                    "threadName": "Payment issue",
                    "createdAt": "2025-02-12T08:00:00+00:00",
                    "ownerId": "14",
                    "archived": True,
                    "locked": False,
                    "messageCount": 4,
                    "memberCount": 2,
                    "totalMessageSent": 5,
                    "rateLimitPerUser": 0,
                    "tags": [],
                    "lastMessageId": None,
                }
            ],
        }
        self.assertEqual(
            result[0].text, json.dumps(expected, ensure_ascii=False, indent=2)
        )

    async def test_list_servers_contract_plain_text(self):
        guild = FakeGuild(
            id=100,
            name="Acme",
            channels=[],
            text_channels=[],
            member_count=3,
            created_at=dt("2024-01-01T00:00:00"),
        )
        fake_client = FakeDiscordClient(guilds=[guild])

        with patch("discord_mcp.server.discord_client", fake_client):
            result = await server.call_tool("list_servers", {})

        self.assertEqual(
            result[0].text, "Available Servers (1):\nAcme (ID: 100, Members: 3)"
        )


class _FakeGuild:
    def __init__(self, guild_id: int, name: str):
        self.id = guild_id
        self.name = name


class _FakeClient:
    def __init__(self, guilds):
        self.guilds = guilds

    def get_guild(self, guild_id: int):
        for guild in self.guilds:
            if guild.id == guild_id:
                return guild
        return None

    async def fetch_guild(self, guild_id: int):
        return self.get_guild(guild_id)


class _FakeForum:
    def __init__(self, active_threads, archived_threads):
        self.threads = active_threads
        self._archived_threads = archived_threads

    async def archived_threads(self, limit=1000):
        for thread in self._archived_threads[:limit]:
            yield thread


class CoreHelperContractTests(unittest.IsolatedAsyncioTestCase):
    def test_try_int_and_normalize_name(self):
        self.assertEqual(_try_int("42"), 42)
        self.assertIsNone(_try_int("not-a-number"))
        self.assertEqual(_normalize_name(" #General "), "general")

    async def test_resolve_guild_raises_on_ambiguous_name(self):
        set_discord_client(_FakeClient([_FakeGuild(1, "Foo"), _FakeGuild(2, "foo")]))
        with self.assertRaisesRegex(ValueError, "Multiple servers found for 'foo'"):
            await _resolve_guild("foo")

    async def test_collect_forum_threads_includes_archived_and_deduplicates(self):
        os.environ["DISCORD_FORUM_MAX_FETCH"] = "1000"
        active = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        archived = [SimpleNamespace(id=2), SimpleNamespace(id=3)]
        forum = _FakeForum(active, archived)

        threads = await _collect_forum_threads(forum, include_archived=True)
        self.assertEqual({thread.id for thread in threads}, {1, 2, 3})

    def test_serialize_message_shape(self):
        attachment = SimpleNamespace(
            id=11,
            filename="file.png",
            url="https://cdn/file.png",
            proxy_url="https://proxy/file.png",
            size=99,
            content_type="image/png",
            width=120,
            height=80,
        )
        embed = SimpleNamespace(
            title="Embed title",
            description="Embed description",
            url="https://example.com",
            image=SimpleNamespace(url="https://img"),
            thumbnail=SimpleNamespace(url="https://thumb"),
        )
        message = SimpleNamespace(
            id=100,
            author="author#0001",
            content="hello",
            created_at=datetime(2025, 1, 1, 0, 0, 0),
            attachments=[attachment],
            embeds=[embed],
        )

        payload = _serialize_message(message)
        self.assertEqual(payload["messageId"], "100")
        self.assertEqual(payload["author"], "author#0001")
        self.assertEqual(payload["attachments"][0]["name"], "file.png")
        self.assertEqual(payload["embeds"][0]["title"], "Embed title")


if __name__ == "__main__":
    unittest.main()
