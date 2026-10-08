import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import discord  # noqa: E402

from discord_mcp.tools.handlers.thread_management import (  # noqa: E402
    handle_add_thread_member,
    handle_create_thread,
    handle_delete_thread,
    handle_edit_thread,
    handle_join_thread,
    handle_leave_thread,
    handle_list_active_threads,
    handle_remove_thread_member,
)
from discord_mcp.tools.schemas.thread_management import (  # noqa: E402
    THREAD_MANAGEMENT_TOOLS,
)

EXPECTED_TOOL_NAMES = [
    "create_thread",
    "join_thread",
    "leave_thread",
    "add_thread_member",
    "remove_thread_member",
    "edit_thread",
    "delete_thread",
    "list_active_threads",
]
GATED_TOOLS = {
    "create_thread",
    "add_thread_member",
    "remove_thread_member",
    "edit_thread",
    "delete_thread",
    "leave_thread",
}


class FakeThread:
    def __init__(self, thread_id, name="release-notes"):
        self.id = int(thread_id)
        self.name = name
        self.type = discord.ChannelType.public_thread
        self.category_id = None
        self.position = 0
        self.nsfw = False
        self.archived = False
        self.locked = False
        self.invitable = True
        self.slowmode_delay = 0
        self.join_count = 0
        self.leave_count = 0
        self.added = []
        self.removed = []
        self.deleted_with = []
        self.edit_kwargs = None

    async def join(self):
        self.join_count += 1

    async def leave(self):
        self.leave_count += 1

    async def add_user(self, user):
        self.added.append(user)

    async def remove_user(self, user):
        self.removed.append(user)

    async def delete(self, *, reason=None):
        self.deleted_with.append(reason)

    async def edit(self, **kwargs):
        self.edit_kwargs = dict(kwargs)
        for key, value in kwargs.items():
            if key in (
                "name",
                "archived",
                "locked",
                "invitable",
                "slowmode_delay",
                "auto_archive_duration",
            ):
                setattr(self, key, value)
        return self


class FakeTextChannel:
    def __init__(self, channel_id, name="general"):
        self.id = int(channel_id)
        self.name = name
        self.type = discord.ChannelType.text
        self.create_kwargs = None

    async def create_thread(self, **kwargs):
        self.create_kwargs = dict(kwargs)
        return FakeThread(555000, name=kwargs["name"])


class FakeForumChannel:
    def __init__(self, channel_id, name="showcase"):
        self.id = int(channel_id)
        self.name = name
        self.type = discord.ChannelType.forum

    async def create_thread(self, **kwargs):
        raise AssertionError("forum parent must be rejected before create_thread")


class FakeMember:
    def __init__(self, member_id, name="alice"):
        self.id = int(member_id)
        self.name = name
        self.display_name = name


class FakeGuild:
    def __init__(self, channels=(), guild_id=9001, name="TestServer"):
        self.id = guild_id
        self.name = name
        self.channels = list(channels)
        self.active = []

    def get_channel(self, channel_id):
        for channel in self.channels:
            if getattr(channel, "id", None) == channel_id:
                return channel
        return None

    async def active_threads(self):
        return list(self.active)


class FakeGateway:
    def __init__(self, guild, threads=None, members=None):
        self.guild = guild
        self.threads = threads or {}
        self.members = members or {}

    async def resolve_guild(self, server_id):
        return self.guild

    async def resolve_thread(self, thread_id, server_id=None):
        thread = self.threads.get(int(thread_id))
        if thread is None:
            raise ValueError(f"Thread '{thread_id}' not found in '{self.guild.name}'")
        return thread, self.guild

    async def resolve_member(self, user_id, server_id=None):
        member = self.members.get(int(user_id))
        if member is None:
            raise ValueError(f"User '{user_id}' not found in '{self.guild.name}'")
        return member


class ThreadManagementSchemaTests(unittest.TestCase):
    def test_eight_tools_matching_the_contract(self):
        self.assertEqual(len(THREAD_MANAGEMENT_TOOLS), 8)
        self.assertEqual(
            [tool.name for tool in THREAD_MANAGEMENT_TOOLS], EXPECTED_TOOL_NAMES
        )
        for tool in THREAD_MANAGEMENT_TOOLS:
            properties = tool.input_schema["properties"]
            if tool.name in GATED_TOOLS:
                self.assertIn("dry_run", properties, tool.name)
                self.assertIn("confirm_token", properties, tool.name)
            else:
                self.assertNotIn("dry_run", properties, tool.name)
                self.assertNotIn("confirm_token", properties, tool.name)

    def test_delete_thread_requires_reason(self):
        delete = next(
            tool for tool in THREAD_MANAGEMENT_TOOLS if tool.name == "delete_thread"
        )
        self.assertIn("reason", delete.input_schema["required"])


class ThreadManagementHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.thread = FakeThread(222, name="release-notes")
        self.text_channel = FakeTextChannel(111, name="general")
        self.forum = FakeForumChannel(333, name="showcase")
        self.guild = FakeGuild(channels=[self.text_channel, self.forum])
        self.gateway = FakeGateway(
            self.guild,
            threads={222: self.thread},
            members={444: FakeMember(444)},
        )
        self.deps = {"gateway": self.gateway}

    async def _call(self, handler, arguments, deps=None):
        result = await handler(arguments, self.deps if deps is None else deps)
        return json.loads(result[0].text)

    async def _execute(self, handler, arguments):
        dry = await self._call(handler, {**arguments, "dry_run": True})
        token = dry["confirmToken"]
        return await self._call(
            handler,
            {**arguments, "dry_run": False, "confirm_token": token},
        )

    async def test_every_handler_requires_gateway(self):
        cases = [
            (handle_create_thread, {"server_id": "1", "channel_id": "111", "name": "x"}),
            (handle_join_thread, {"server_id": "1", "thread_id": "222"}),
            (handle_leave_thread, {"server_id": "1", "thread_id": "222"}),
            (
                handle_add_thread_member,
                {"server_id": "1", "thread_id": "222", "user_id": "444"},
            ),
            (
                handle_remove_thread_member,
                {"server_id": "1", "thread_id": "222", "user_id": "444"},
            ),
            (handle_edit_thread, {"server_id": "1", "thread_id": "222", "name": "y"}),
            (
                handle_delete_thread,
                {"server_id": "1", "thread_id": "222", "reason": "cleanup"},
            ),
            (handle_list_active_threads, {"server_id": "1"}),
        ]
        for handler, arguments in cases:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "gateway is required"):
                    await handler(arguments, {})

    async def test_create_thread_dry_run_returns_confirm_token(self):
        payload = await self._call(
            handle_create_thread,
            {"server_id": "1", "channel_id": "111", "name": "release"},
        )
        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["action"], "create_thread")
        self.assertTrue(payload["confirmToken"])
        self.assertIsNone(self.text_channel.create_kwargs)

    async def test_create_thread_execute_without_token_raises(self):
        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await self._call(
                handle_create_thread,
                {
                    "server_id": "1",
                    "channel_id": "111",
                    "name": "release",
                    "dry_run": False,
                },
            )
        self.assertIsNone(self.text_channel.create_kwargs)

    async def test_create_thread_executes_with_token_and_payload_keys(self):
        payload = await self._execute(
            handle_create_thread,
            {
                "server_id": "1",
                "channel_id": "111",
                "name": "release",
                "type": "public_thread",
                "auto_archive_duration": 1440,
                "slowmode_delay": 5,
                "reason": "prep",
            },
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "create_thread")
        self.assertIn("thread", payload)
        self.assertEqual(payload["thread"]["id"], "555000")
        self.assertEqual(payload["thread"]["name"], "release")
        self.assertEqual(
            self.text_channel.create_kwargs["type"],
            discord.ChannelType.public_thread,
        )
        self.assertEqual(
            self.text_channel.create_kwargs["auto_archive_duration"], 1440
        )
        self.assertEqual(self.text_channel.create_kwargs["slowmode_delay"], 5)
        self.assertEqual(self.text_channel.create_kwargs["reason"], "prep")

    async def test_create_thread_rejects_forum_parent(self):
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_create_thread,
                {"server_id": "1", "channel_id": "333", "name": "hello"},
            )
        message = str(ctx.exception)
        self.assertIn("showcase", message)
        self.assertIn("TestServer", message)
        self.assertIn("forum-post", message)

    async def test_create_thread_rejects_invalid_auto_archive_duration(self):
        with self.assertRaisesRegex(
            ValueError, "auto_archive_duration must be one of: 60, 1440, 4320, 10080"
        ):
            await self._call(
                handle_create_thread,
                {
                    "server_id": "1",
                    "channel_id": "111",
                    "name": "x",
                    "auto_archive_duration": 100,
                },
            )

    async def test_create_thread_rejects_out_of_range_slowmode(self):
        with self.assertRaisesRegex(ValueError, "slowmode_delay"):
            await self._call(
                handle_create_thread,
                {
                    "server_id": "1",
                    "channel_id": "111",
                    "name": "x",
                    "slowmode_delay": 21601,
                },
            )

    async def test_join_and_leave_thread_run_without_gate(self):
        join = await self._call(
            handle_join_thread, {"server_id": "1", "thread_id": "222"}
        )
        self.assertEqual(
            join,
            {"status": "executed", "action": "join_thread", "threadId": "222"},
        )
        self.assertEqual(self.thread.join_count, 1)

        dry = await self._call(
            handle_leave_thread, {"server_id": "1", "thread_id": "222"}
        )
        self.assertEqual(dry["status"], "dry_run")
        self.assertTrue(dry["confirmToken"])
        self.assertEqual(self.thread.leave_count, 0)

        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await self._call(
                handle_leave_thread,
                {"server_id": "1", "thread_id": "222", "dry_run": False},
            )

        leave = await self._call(
            handle_leave_thread,
            {
                "server_id": "1",
                "thread_id": "222",
                "dry_run": False,
                "confirm_token": dry["confirmToken"],
            },
        )
        self.assertEqual(
            leave,
            {"status": "executed", "action": "leave_thread", "threadId": "222"},
        )
        self.assertEqual(self.thread.leave_count, 1)

    async def test_add_thread_member_gated_payload_keys(self):
        base = {"server_id": "1", "thread_id": "222", "user_id": "444"}
        dry = await self._call(handle_add_thread_member, base)
        self.assertEqual(dry["status"], "dry_run")
        self.assertTrue(dry["confirmToken"])
        self.assertEqual(self.thread.added, [])

        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await self._call(handle_add_thread_member, {**base, "dry_run": False})

        payload = await self._execute(handle_add_thread_member, base)
        self.assertEqual(
            payload,
            {
                "status": "executed",
                "action": "add_thread_member",
                "threadId": "222",
                "userId": "444",
            },
        )
        self.assertEqual(len(self.thread.added), 1)

    async def test_remove_thread_member_executes_with_token(self):
        base = {"server_id": "1", "thread_id": "222", "user_id": "444"}
        payload = await self._execute(handle_remove_thread_member, base)
        self.assertEqual(
            payload,
            {
                "status": "executed",
                "action": "remove_thread_member",
                "threadId": "222",
                "userId": "444",
            },
        )
        self.assertEqual(len(self.thread.removed), 1)

    async def test_edit_thread_payload_keys(self):
        base = {
            "server_id": "1",
            "thread_id": "222",
            "name": "renamed",
            "archived": False,
            "locked": True,
            "invitable": False,
            "slowmode_delay": 10,
        }
        payload = await self._execute(handle_edit_thread, base)
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "edit_thread")
        for key in ("threadId", "name", "archived", "locked", "invitable", "slowmodeDelay"):
            self.assertIn(key, payload)
        self.assertEqual(payload["threadId"], "222")
        self.assertEqual(payload["name"], "renamed")
        self.assertEqual(payload["slowmodeDelay"], 10)
        self.assertEqual(self.thread.edit_kwargs["name"], "renamed")
        self.assertEqual(self.thread.edit_kwargs["slowmode_delay"], 10)

    async def test_edit_thread_without_any_field_raises(self):
        with self.assertRaisesRegex(ValueError, "at least one of"):
            await self._call(
                handle_edit_thread, {"server_id": "1", "thread_id": "222"}
            )

    async def test_delete_thread_requires_reason_even_for_dry_run(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            await self._call(
                handle_delete_thread, {"server_id": "1", "thread_id": "222"}
            )

    async def test_delete_thread_executes_with_token(self):
        payload = await self._execute(
            handle_delete_thread,
            {"server_id": "1", "thread_id": "222", "reason": "cleanup"},
        )
        self.assertEqual(
            payload,
            {
                "status": "executed",
                "action": "delete_thread",
                "threadId": "222",
            },
        )
        self.assertEqual(self.thread.deleted_with, ["cleanup"])

    async def test_list_active_threads_payload(self):
        other = FakeThread(777, name="help")
        self.guild.active = [self.thread, other]

        payload = await self._call(handle_list_active_threads, {"server_id": "1"})
        self.assertEqual(payload["serverId"], "9001")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(
            [row["id"] for row in payload["threads"]], ["222", "777"]
        )
        self.assertIn("archived", payload["threads"][0])


if __name__ == "__main__":
    unittest.main()
