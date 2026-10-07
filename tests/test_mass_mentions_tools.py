import json
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

import discord  # noqa: E402
from discord_mcp.tools.handlers.mass_mentions import (  # noqa: E402
    handle_audit_mass_mentions,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER  # noqa: E402
from discord_mcp.tools.schemas import compose_tool_registry  # noqa: E402


MENTION_EVERYONE = 1 << 17
SEND_MESSAGES = 1 << 11


class FakeAuthor:
    def __init__(self, author_id, name):
        self.id = author_id
        self.name = name
        self.display_name = name

    def __str__(self):
        return self.name


class FakeMessage:
    def __init__(self, message_id, author, content, created_at, mention_everyone=False):
        self.id = message_id
        self.author = author
        self.content = content
        self.created_at = created_at
        self.mention_everyone = mention_everyone
        self.mentions = []
        self.role_mentions = []


class FakeRole:
    def __init__(self, role_id, name, position=0, permissions=0):
        self.id = role_id
        self.name = name
        self.position = position
        self.permissions = discord.Permissions(permissions)
        self.hoist = False
        self.mentionable = False
        self.managed = False


class FakeMember:
    def __init__(self, member_id, name, roles):
        self.id = member_id
        self.name = name
        self.display_name = name
        self.roles = roles


class FakeChannel:
    def __init__(
        self,
        channel_id,
        name,
        messages,
        history_error=None,
        threads=None,
        channel_type="text",
        archived=None,
    ):
        self.id = channel_id
        self.name = name
        self.type = channel_type
        self.category_id = None
        self.category = None
        self.parent = None
        self.overwrites = {}
        self._messages = messages
        self._history_error = history_error
        self.threads = threads or []
        self._archived = archived or []

    def history(self, limit=200, before=None):
        messages = self._messages[:limit]
        error = self._history_error

        async def iterator():
            if error is not None:
                raise error
            for message in messages:
                yield message

        return iterator()

    def archived_threads(self, limit=100):
        archived = self._archived

        async def iterator():
            for thread in archived:
                yield thread

        return iterator()


class FakeGuild:
    def __init__(self, channels, members):
        self.id = 1
        self.name = "Guild"
        self.channels = channels
        self.members = members

    def get_member(self, member_id):
        return self.members.get(member_id)


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, _server_id):
        return self.guild


def _dt(hours_ago):
    return datetime.now(timezone.utc) - timedelta(hours=hours_ago)


class MassMentionRegistryTests(unittest.TestCase):
    def test_tool_registered_in_schema_and_router(self):
        names = {tool.name for tool in compose_tool_registry()}
        self.assertIn("audit_mass_mentions", names)
        self.assertIn("audit_mass_mentions", TOOL_ROUTER)


class MassMentionHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.granting_role = FakeRole(2, "🍀 Thành viên", 5, MENTION_EVERYONE)
        self.plain_role = FakeRole(3, "Member", 1, SEND_MESSAGES)
        self.sender = FakeMember(10, "shiro", [self.plain_role])
        self.granter = FakeMember(11, "admin", [self.plain_role, self.granting_role])
        self.guild = FakeGuild(
            channels=[
                FakeChannel(
                    100,
                    "chat",
                    [
                        FakeMessage(1, FakeAuthor(10, "shiro"), "@everyone", _dt(1)),
                        FakeMessage(
                            2,
                            FakeAuthor(11, "admin"),
                            "ping @everyone now",
                            _dt(2),
                            mention_everyone=True,
                        ),
                        FakeMessage(
                            3, FakeAuthor(11, "admin"), "welcome @here", _dt(3)
                        ),
                        FakeMessage(4, FakeAuthor(10, "shiro"), "hello", _dt(4)),
                        FakeMessage(5, FakeAuthor(10, "shiro"), "@everyone", _dt(400)),
                    ],
                ),
                FakeChannel(200, "voice", []),
                FakeChannel(
                    300, "private", [], history_error=AttributeError("no history")
                ),
            ],
            members={10: self.sender, 11: self.granter},
        )
        self.deps = {"gateway": FakeGateway(self.guild)}

    async def test_delivered_and_suppressed_hits_are_separated(self):
        result = await handle_audit_mass_mentions(
            {"server_id": "1", "scan_limit": 100, "window_hours": 0}, self.deps
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["hitCount"], 4)
        self.assertEqual(payload["deliveredCount"], 1)
        self.assertEqual(payload["suppressedCount"], 3)

        delivered = [h for h in payload["hits"] if h["kind"] == "delivered"]
        self.assertEqual(delivered[0]["messageId"], "2")
        self.assertTrue(delivered[0]["mentionEveryone"])
        self.assertEqual(delivered[0]["authorId"], "11")
        self.assertTrue(delivered[0]["authorHasMentionEveryoneNow"])
        self.assertEqual(
            delivered[0]["authorGrantingRoles"][0]["name"], "🍀 Thành viên"
        )

    async def test_literal_everyone_text_is_not_a_ping(self):
        result = await handle_audit_mass_mentions(
            {"server_id": "1", "scan_limit": 100, "window_hours": 0}, self.deps
        )
        payload = json.loads(result[0].text)

        suppressed = {
            h["messageId"]: h for h in payload["hits"] if h["kind"] == "suppressed_text"
        }
        self.assertEqual(sorted(suppressed), ["1", "3", "5"])
        target = suppressed["1"]
        self.assertFalse(target["mentionEveryone"])
        self.assertEqual(target["content"], "@everyone")
        self.assertFalse(target["authorHasMentionEveryoneNow"])
        self.assertEqual(target["authorGrantingRoles"], [])

    async def test_window_and_channel_filter(self):
        result = await handle_audit_mass_mentions(
            {"server_id": "1", "scan_limit": 100, "window_hours": 24}, self.deps
        )
        payload = json.loads(result[0].text)
        self.assertEqual([h["messageId"] for h in payload["hits"]], ["3", "2", "1"])

        filtered = await handle_audit_mass_mentions(
            {"server_id": "1", "channel_ids": ["200"], "window_hours": 0}, self.deps
        )
        filtered_payload = json.loads(filtered[0].text)
        self.assertEqual(filtered_payload["hitCount"], 0)
        self.assertEqual(filtered_payload["accessErrors"], [])

        unreadable = await handle_audit_mass_mentions(
            {"server_id": "1", "channel_ids": ["300"], "window_hours": 0}, self.deps
        )
        unreadable_payload = json.loads(unreadable[0].text)
        self.assertEqual(unreadable_payload["accessErrors"][0]["channelId"], "300")
        self.assertEqual(unreadable_payload["hitCount"], 0)

    async def test_forum_posts_are_scanned_through_threads(self):
        forum = FakeChannel(
            400,
            "forum",
            [],
            channel_type="forum",
            archived=[
                FakeChannel(
                    401,
                    "post",
                    [
                        FakeMessage(
                            9,
                            FakeAuthor(11, "admin"),
                            "@everyone read this",
                            _dt(2),
                            mention_everyone=True,
                        )
                    ],
                    channel_type="public_thread",
                )
            ],
        )
        self.guild.channels.append(forum)
        self.deps = {"gateway": FakeGateway(self.guild)}

        result = await handle_audit_mass_mentions(
            {
                "server_id": "1",
                "channel_ids": ["400"],
                "window_hours": 0,
                "include_threads": True,
            },
            self.deps,
        )
        payload = json.loads(result[0].text)

        self.assertEqual(payload["accessErrors"], [])
        self.assertEqual(payload["hitCount"], 1)
        self.assertEqual(payload["hits"][0]["messageId"], "9")
        self.assertEqual(payload["hits"][0]["kind"], "delivered")
        self.assertEqual(payload["hits"][0]["channelId"], "401")

        without_threads = await handle_audit_mass_mentions(
            {"server_id": "1", "channel_ids": ["400"], "window_hours": 0}, self.deps
        )
        self.assertEqual(json.loads(without_threads[0].text)["hitCount"], 0)

    async def test_clean_channels_report_zero_hits(self):
        result = await handle_audit_mass_mentions(
            {"server_id": "1", "channel_ids": ["100"], "window_hours": 1}, self.deps
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["scannedChannels"], 1)
        self.assertGreaterEqual(payload["scannedMessages"], 1)
        self.assertIn("mention_everyone", payload["note"])


if __name__ == "__main__":
    unittest.main()
