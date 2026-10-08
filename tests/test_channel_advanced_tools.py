import json
import os
import sys
import unittest
from unittest.mock import AsyncMock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import discord  # noqa: E402

from discord_mcp.core.permissions import as_permission_bits  # noqa: E402
from discord_mcp.tools.handlers import channel_advanced as tools  # noqa: E402
from discord_mcp.tools.schemas.channel_advanced import (  # noqa: E402
    CHANNEL_ADVANCED_TOOLS,
)

SEND_MESSAGES = 1 << 11
VIEW_CHANNEL = 1 << 10


def ow(allow: int, deny: int) -> discord.PermissionOverwrite:
    return discord.PermissionOverwrite.from_pair(
        discord.Permissions(allow), discord.Permissions(deny)
    )


class FakeRole:
    def __init__(self, role_id, name):
        self.id = int(role_id)
        self.name = name


class FakeWebhook:
    def __init__(self, webhook_id, url):
        self.id = webhook_id
        self.url = url


class FakeChannel:
    def __init__(
        self,
        channel_id,
        name,
        type_="text",
        category_id=None,
        position=0,
        overwrites=None,
        news=False,
    ):
        self.id = int(channel_id)
        self.name = name
        self.type = type_
        self.category_id = category_id
        self.position = position
        self.overwrites = dict(overwrites or {})
        self._news = news
        self.clone = AsyncMock()
        self.follow = AsyncMock()
        self.set_permissions = AsyncMock()
        self.edit = AsyncMock()

    def is_news(self):
        return self._news


class FakeGuild:
    def __init__(self, channels=()):
        self.name = "Test Guild"
        self.channels = list(channels)
        self.create_text_channel = AsyncMock()
        self.create_stage_channel = AsyncMock()

    def get_channel(self, channel_id):
        for channel in self.channels:
            if channel.id == channel_id:
                return channel
        return None


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, server_id):
        return self.guild


class ChannelAdvancedTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.role_mods = FakeRole(2, "Mods")
        self.role_other = FakeRole(3, "Other")
        self.category = FakeChannel(
            100,
            "General",
            type_="category",
            overwrites={self.role_mods: ow(SEND_MESSAGES, 0)},
        )
        self.chat = FakeChannel(
            101,
            "chat",
            type_="text",
            category_id=100,
            overwrites={self.role_mods: ow(SEND_MESSAGES, 0)},
        )
        self.news = FakeChannel(
            102, "announcements", type_="news", category_id=100, news=True
        )
        self.voice = FakeChannel(
            103,
            "Lounge",
            type_="voice",
            category_id=100,
            overwrites={self.role_other: ow(0, VIEW_CHANNEL)},
        )
        self.stage = FakeChannel(104, "Stage", type_="stage", category_id=100)
        self.guild = FakeGuild(
            [self.category, self.chat, self.news, self.voice, self.stage]
        )
        self.gateway = FakeGateway(self.guild)

    def deps(self):
        return {"gateway": self.gateway}

    def cases(self):
        return [
            (
                "clone_channel",
                tools.handle_clone_channel,
                {"server_id": "1", "channel_id": "101"},
            ),
            (
                "create_announcement_channel",
                tools.handle_create_announcement_channel,
                {"server_id": "1", "name": "news-room"},
            ),
            (
                "create_stage_channel",
                tools.handle_create_stage_channel,
                {"server_id": "1", "name": "stage-room"},
            ),
            (
                "follow_channel",
                tools.handle_follow_channel,
                {"server_id": "1", "channel_id": "102", "webhook_channel_id": "101"},
            ),
            (
                "sync_channel_permissions",
                tools.handle_sync_channel_permissions,
                {"server_id": "1", "category_id": "100"},
            ),
            (
                "set_voice_channel_status",
                tools.handle_set_voice_channel_status,
                {"server_id": "1", "channel_id": "103", "status": "Recording"},
            ),
        ]

    async def dry_run_token(self, handler, args):
        result = await handler({**args, "dry_run": True}, self.deps())
        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "dry_run")
        self.assertTrue(payload["confirmToken"])
        return payload["confirmToken"]


class SchemaTests(ChannelAdvancedTestCase):
    def test_exactly_six_tools_in_table_order(self):
        names = [tool.name for tool in CHANNEL_ADVANCED_TOOLS]
        self.assertEqual(
            names,
            [
                "clone_channel",
                "create_announcement_channel",
                "create_stage_channel",
                "follow_channel",
                "sync_channel_permissions",
                "set_voice_channel_status",
            ],
        )

    def test_every_tool_declares_gate_params(self):
        for tool in CHANNEL_ADVANCED_TOOLS:
            with self.subTest(tool=tool.name):
                schema = tool.input_schema
                self.assertEqual(schema["type"], "object")
                self.assertTrue(schema["required"])
                self.assertIn("dry_run", schema["properties"])
                self.assertIn("confirm_token", schema["properties"])

    def test_sync_description_documents_client_side_mirror(self):
        sync = next(
            tool
            for tool in CHANNEL_ADVANCED_TOOLS
            if tool.name == "sync_channel_permissions"
        )
        self.assertIn("client-side", sync.description)


class GateContractTests(ChannelAdvancedTestCase):
    async def test_every_handler_requires_gateway(self):
        for name, handler, args in self.cases():
            with self.subTest(tool=name):
                with self.assertRaisesRegex(
                    ValueError, f"gateway is required for {name}"
                ):
                    await handler(args, {})

    async def test_dry_run_returns_token_and_execute_requires_it(self):
        for name, handler, args in self.cases():
            with self.subTest(tool=name):
                token = await self.dry_run_token(handler, args)
                with self.assertRaisesRegex(ValueError, "confirm_token is required"):
                    await handler(
                        {**args, "dry_run": False, "confirm_token": ""}, self.deps()
                    )
                with self.assertRaisesRegex(ValueError, "Invalid confirm_token"):
                    await handler(
                        {**args, "dry_run": False, "confirm_token": "bad"},
                        self.deps(),
                    )
                self.assertTrue(token)


class CloneChannelTests(ChannelAdvancedTestCase):
    async def test_executed_payload_and_default_name(self):
        token = await self.dry_run_token(
            tools.handle_clone_channel, {"server_id": "1", "channel_id": "101"}
        )
        self.chat.clone.return_value = FakeChannel(900, "chat-copy")

        result = await tools.handle_clone_channel(
            {
                "server_id": "1",
                "channel_id": "101",
                "dry_run": False,
                "confirm_token": token,
            },
            self.deps(),
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "clone_channel")
        self.assertEqual(payload["channel"]["id"], "900")
        self.chat.clone.assert_awaited_once_with(name="chat", reason=None)

    async def test_non_clonable_channel_rejected(self):
        self.chat.clone = None
        with self.assertRaisesRegex(ValueError, "does not support cloning"):
            await tools.handle_clone_channel(
                {"server_id": "1", "channel_id": "101"}, self.deps()
            )


class CreateChannelTests(ChannelAdvancedTestCase):
    async def test_announcement_payload_and_kwargs(self):
        args = {
            "server_id": "1",
            "name": "news-room",
            "category_id": "100",
            "topic": "alerts",
            "nsfw": False,
        }
        token = await self.dry_run_token(tools.handle_create_announcement_channel, args)
        self.guild.create_text_channel.return_value = FakeChannel(
            901, "news-room", type_="news", category_id=100
        )

        result = await tools.handle_create_announcement_channel(
            {**args, "dry_run": False, "confirm_token": token}, self.deps()
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "create_announcement_channel")
        self.assertEqual(payload["channel"]["id"], "901")
        kwargs = self.guild.create_text_channel.call_args.kwargs
        self.assertTrue(kwargs["news"])
        self.assertIs(kwargs["category"], self.category)
        self.assertEqual(kwargs["topic"], "alerts")
        self.assertIs(kwargs["nsfw"], False)
        self.assertNotIn("position", kwargs)

    async def test_stage_payload_and_video_quality_mode(self):
        args = {
            "server_id": "1",
            "name": "stage-room",
            "category_id": "100",
            "bitrate": 64000,
            "video_quality_mode": 2,
        }
        token = await self.dry_run_token(tools.handle_create_stage_channel, args)
        self.guild.create_stage_channel.return_value = FakeChannel(
            902, "stage-room", type_="stage", category_id=100
        )

        result = await tools.handle_create_stage_channel(
            {**args, "dry_run": False, "confirm_token": token}, self.deps()
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "create_stage_channel")
        self.assertEqual(payload["channel"]["id"], "902")
        kwargs = self.guild.create_stage_channel.call_args.kwargs
        self.assertIs(kwargs["video_quality_mode"], discord.VideoQualityMode.full)
        self.assertEqual(kwargs["bitrate"], 64000)

    async def test_invalid_video_quality_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "video_quality_mode must be 1"):
            await tools.handle_create_stage_channel(
                {"server_id": "1", "name": "stage-room", "video_quality_mode": 9},
                self.deps(),
            )

    async def test_category_not_found(self):
        with self.assertRaisesRegex(
            ValueError, "Category '999' not found in server 'Test Guild'"
        ):
            await tools.handle_create_announcement_channel(
                {"server_id": "1", "name": "news-room", "category_id": "999"},
                self.deps(),
            )

    async def test_category_id_must_be_a_category(self):
        with self.assertRaisesRegex(ValueError, "not a category"):
            await tools.handle_create_announcement_channel(
                {"server_id": "1", "name": "news-room", "category_id": "101"},
                self.deps(),
            )


class FollowChannelTests(ChannelAdvancedTestCase):
    async def test_executed_payload(self):
        args = {"server_id": "1", "channel_id": "102", "webhook_channel_id": "101"}
        token = await self.dry_run_token(tools.handle_follow_channel, args)
        self.news.follow.return_value = FakeWebhook(
            555, "https://discord.com/api/webhooks/555/xyz"
        )

        result = await tools.handle_follow_channel(
            {**args, "dry_run": False, "confirm_token": token}, self.deps()
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "follow_channel")
        self.assertEqual(payload["webhookId"], "555")
        self.assertEqual(
            payload["webhookUrl"], "https://discord.com/api/webhooks/555/xyz"
        )
        self.news.follow.assert_awaited_once_with(destination=self.chat)

    async def test_source_must_be_news(self):
        with self.assertRaisesRegex(ValueError, "announcement"):
            await tools.handle_follow_channel(
                {"server_id": "1", "channel_id": "101", "webhook_channel_id": "102"},
                self.deps(),
            )

    async def test_destination_must_be_text(self):
        with self.assertRaisesRegex(ValueError, "must be a text channel"):
            await tools.handle_follow_channel(
                {"server_id": "1", "channel_id": "102", "webhook_channel_id": "103"},
                self.deps(),
            )


class SyncChannelPermissionsTests(ChannelAdvancedTestCase):
    async def test_dry_run_reports_per_channel_diff_without_writing(self):
        result = await tools.handle_sync_channel_permissions(
            {"server_id": "1", "category_id": "100"}, self.deps()
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["action"], "sync_channel_permissions")
        self.assertTrue(payload["confirmToken"])
        diffs = payload["details"]["diffs"]
        self.assertEqual(
            [entry["channelId"] for entry in diffs], ["101", "102", "103", "104"]
        )
        by_id = {entry["channelId"]: entry for entry in diffs}
        self.assertFalse(by_id["101"]["changed"])  # already mirrors the category
        self.assertTrue(by_id["102"]["changed"])
        for entry in diffs:
            self.assertEqual(
                sorted(entry),
                ["applied", "changed", "channelId", "channelName", "previous"],
            )
        # previous shows the child's extra overwrite, applied shows the category's
        self.assertIn("3", by_id["103"]["previous"])
        self.assertIn("2", by_id["103"]["applied"])
        for child in (self.chat, self.news, self.voice, self.stage):
            child.set_permissions.assert_not_called()

    async def test_execute_writes_only_changed_targets(self):
        token = await self.dry_run_token(
            tools.handle_sync_channel_permissions,
            {"server_id": "1", "category_id": "100"},
        )

        result = await tools.handle_sync_channel_permissions(
            {
                "server_id": "1",
                "category_id": "100",
                "dry_run": False,
                "confirm_token": token,
            },
            self.deps(),
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "sync_channel_permissions")
        self.assertEqual(payload["categoryId"], "100")
        self.assertEqual(payload["updated"], ["102", "103", "104"])
        self.assertEqual(payload["unchanged"], 1)

        self.chat.set_permissions.assert_not_called()

        args, kwargs = self.news.set_permissions.call_args
        self.assertIs(args[0], self.role_mods)
        allow, deny = kwargs["overwrite"].pair()
        self.assertEqual(
            (as_permission_bits(allow), as_permission_bits(deny)),
            (SEND_MESSAGES, 0),
        )

        # voice keeps an overwrite the category lacks -> deleted
        self.assertEqual(self.voice.set_permissions.call_count, 2)
        del_args, del_kwargs = self.voice.set_permissions.call_args
        self.assertIs(del_args[0], self.role_other)
        self.assertIsNone(del_kwargs["overwrite"])

    async def test_category_with_no_children(self):
        lone = FakeGuild([FakeChannel(200, "Solo", type_="category")])
        gateway = FakeGateway(lone)

        result = await tools.handle_sync_channel_permissions(
            {"server_id": "1", "category_id": "200"}, {"gateway": gateway}
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["details"]["diffs"], [])
        token = payload["confirmToken"]

        executed = await tools.handle_sync_channel_permissions(
            {
                "server_id": "1",
                "category_id": "200",
                "dry_run": False,
                "confirm_token": token,
            },
            {"gateway": gateway},
        )
        done = json.loads(executed[0].text)
        self.assertEqual(done["status"], "executed")
        self.assertEqual(done["updated"], [])
        self.assertEqual(done["unchanged"], 0)


class SetVoiceChannelStatusTests(ChannelAdvancedTestCase):
    async def test_executed_payload(self):
        args = {"server_id": "1", "channel_id": "103", "status": "Recording"}
        token = await self.dry_run_token(tools.handle_set_voice_channel_status, args)

        result = await tools.handle_set_voice_channel_status(
            {**args, "dry_run": False, "confirm_token": token}, self.deps()
        )

        payload = json.loads(result[0].text)
        self.assertEqual(
            payload,
            {
                "status": "executed",
                "action": "set_voice_channel_status",
                "channelId": "103",
                "channelStatus": "Recording",
            },
        )
        self.voice.edit.assert_awaited_once_with(status="Recording", reason=None)

    async def test_status_too_long_rejected(self):
        with self.assertRaisesRegex(ValueError, "at most 500 characters"):
            await tools.handle_set_voice_channel_status(
                {"server_id": "1", "channel_id": "103", "status": "x" * 501},
                self.deps(),
            )

    async def test_empty_and_whitespace_status_rejected(self):
        for bad in ("", "   "):
            with self.subTest(status=repr(bad)):
                with self.assertRaisesRegex(ValueError, "1-500 characters"):
                    await tools.handle_set_voice_channel_status(
                        {"server_id": "1", "channel_id": "103", "status": bad},
                        self.deps(),
                    )

    async def test_non_string_status_rejected(self):
        with self.assertRaisesRegex(ValueError, "1-500 characters"):
            await tools.handle_set_voice_channel_status(
                {"server_id": "1", "channel_id": "103", "status": 42},
                self.deps(),
            )

    async def test_wrong_channel_type_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "only supported on voice and stage channels"
        ):
            await tools.handle_set_voice_channel_status(
                {"server_id": "1", "channel_id": "101", "status": "Recording"},
                self.deps(),
            )

    async def test_stage_channel_accepted(self):
        args = {"server_id": "1", "channel_id": "104", "status": "Q and A"}
        token = await self.dry_run_token(tools.handle_set_voice_channel_status, args)

        result = await tools.handle_set_voice_channel_status(
            {**args, "dry_run": False, "confirm_token": token}, self.deps()
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["channelStatus"], "Q and A")
        self.stage.edit.assert_awaited_once_with(status="Q and A", reason=None)


if __name__ == "__main__":
    unittest.main()
