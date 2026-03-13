import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("DISCORD_TOKEN", "test-token")

import discord_mcp.server as server  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
