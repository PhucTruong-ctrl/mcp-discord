import discord
import importlib
import json
import os
import sys
import types
import unittest
from types import SimpleNamespace


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

sys.modules.setdefault("discord", types.ModuleType("discord"))
sys.modules["discord"].ForumChannel = type("ForumChannel", (), {})
sys.modules["discord"].TextChannel = type("TextChannel", (), {})
sys.modules["discord"].VoiceChannel = type("VoiceChannel", (), {})
sys.modules["discord"].Intents = SimpleNamespace(
    default=lambda: SimpleNamespace(message_content=False, members=False)
)
discord_ext = types.ModuleType("discord.ext")


class _Bot:
    def __init__(self, *args, **kwargs):
        self.user = SimpleNamespace(name="test-bot")

    def event(self, func):
        return func

    async def start(self, *_args, **_kwargs):
        return None


discord_ext.commands = SimpleNamespace(Bot=_Bot)
sys.modules.setdefault("discord.ext", discord_ext)
sys.modules.setdefault("discord.ext.commands", discord_ext.commands)



aiohttp = types.ModuleType("aiohttp")
aiohttp.ClientSession = type(
    "ClientSession",
    (),
    {
        "__aenter__": lambda self: self,
        "__aexit__": lambda self, exc_type, exc, tb: False,
    },
)
sys.modules.setdefault("aiohttp", aiohttp)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

router = importlib.import_module("discord_mcp.tools.handlers.router")
schemas = importlib.import_module("discord_mcp.tools.schemas")

from discord_mcp.tools.handlers.inventory import (
    handle_get_channel_hierarchy,
    handle_get_channels_structured,
    handle_get_permission_overwrites,
)


from discord_mcp.core.serialize import _serialize_forum_tag
from discord_mcp.tools.handlers.channels import _build_forum_tags, _guard_tag_ids


class ChannelAdminToolRegistryTests(unittest.TestCase):
    def test_channel_admin_tools_are_present_in_schema_registry(self):
        names = [tool.name for tool in schemas.compose_tool_registry()]

        for name in [
            "create_voice_channel",
            "create_forum_channel",
            "update_text_channel",
            "update_voice_channel",
            "update_forum_channel",
        ]:
            self.assertIn(name, names)

    def test_channel_admin_tools_are_registered_in_router(self):
        for tool_name in [
            "create_voice_channel",
            "create-voice-channel",
            "create_forum_channel",
            "create-forum-channel",
            "update_text_channel",
            "update-text-channel",
            "update_voice_channel",
            "update-voice-channel",
            "update_forum_channel",
            "update-forum-channel",
        ]:
            self.assertIn(tool_name, router.TOOL_ROUTER)


class ChannelAdminHandlerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_text_channel_maps_each_field(self):
        handler = router.TOOL_ROUTER["update_text_channel"]
        guild = self._guild(
            channels=[
                self._channel(10, "general", type="text", topic="old", nsfw=False),
            ]
        )
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        result = await handler(
            {
                "server_id": "1",
                "channel_id": "10",
                "name": "announcements",
                "topic": "new topic",
                "nsfw": True,
            },
            {"gateway": gateway},
        )

        self.assertEqual(result[0].type, "text")
        self.assertIn("announcements", result[0].text)
        self.assertEqual(guild.channels[0].edit_calls[-1]["topic"], "new topic")

    async def test_update_voice_channel_maps_each_field(self):
        handler = router.TOOL_ROUTER["update_voice_channel"]
        guild = self._guild(
            channels=[
                self._channel(20, "voice", type="voice", bitrate=64000, user_limit=0),
            ]
        )
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        result = await handler(
            {
                "server_id": "1",
                "channel_id": "20",
                "name": "ops voice",
                "bitrate": 96000,
                "user_limit": 12,
                "rtc_region": "us-east",
            },
            {"gateway": gateway},
        )

        self.assertEqual(result[0].type, "text")
        self.assertIn("ops voice", result[0].text)
        self.assertEqual(guild.channels[0].edit_calls[-1]["bitrate"], 96000)

    async def test_update_forum_channel_maps_each_field(self):
        handler = router.TOOL_ROUTER["update_forum_channel"]
        guild = self._guild(
            channels=[
                self._channel(
                    30, "forum", type="forum", topic="old", available_tags=[]
                ),
            ]
        )
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        result = await handler(
            {
                "server_id": "1",
                "channel_id": "30",
                "name": "knowledge-base",
                "topic": "Forum topics",
                "available_tags": [{"name": "help"}],
            },
            {"gateway": gateway},
        )

        self.assertEqual(result[0].type, "text")
        self.assertIn("knowledge-base", result[0].text)
        self.assertEqual(guild.channels[0].edit_calls[-1]["topic"], "Forum topics")

    async def test_update_text_channel_rejects_wrong_channel_type(self):
        handler = router.TOOL_ROUTER["update_text_channel"]
        guild = self._guild(
            channels=[
                self._channel(40, "voice", type="voice"),
            ]
        )
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        with self.assertRaisesRegex(ValueError, "text channel"):
            await handler(
                {"server_id": "1", "channel_id": "40", "name": "general"},
                {"gateway": gateway},
            )

    async def test_update_voice_channel_requires_identifier(self):
        handler = router.TOOL_ROUTER["update_voice_channel"]
        guild = self._guild(channels=[])
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        with self.assertRaisesRegex((KeyError, ValueError), "channel_id"):
            await handler({"server_id": "1", "name": "voice"}, {"gateway": gateway})

    async def test_update_forum_channel_rejects_unknown_fields(self):
        handler = router.TOOL_ROUTER["update_forum_channel"]
        guild = self._guild(
            channels=[
                self._channel(50, "forum", type="forum", available_tags=[]),
            ]
        )
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        with self.assertRaisesRegex(ValueError, "unsupported_fields"):
            await handler(
                {
                    "server_id": "1",
                    "channel_id": "50",
                    "name": "forum",
                    "unknown_field": True,
                },
                {"gateway": gateway},
            )

    async def test_update_forum_channel_accepts_default_sort_order(self):
        """default_sort_order is supported by discord.py 2.7.1+ ForumChannel.edit()."""
        handler = router.TOOL_ROUTER["update_forum_channel"]
        channel = self._channel(60, "forum", type="forum", available_tags=[])
        guild = self._guild(channels=[channel])
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        result = await handler(
            {
                "server_id": "1",
                "channel_id": "60",
                "name": "forum",
                "default_sort_order": 1,
            },
            {"gateway": gateway},
        )

        self.assertIn("Updated forum channel", result[0].text)

    async def test_create_voice_channel_uses_voice_creator(self):
        handler = router.TOOL_ROUTER["create_voice_channel"]
        guild = self._guild()

        async def create_voice_channel(**kwargs):
            payload = dict(kwargs)
            payload.pop("name", None)
            return self._channel(80, kwargs["name"], type="voice", **payload)

        guild.create_voice_channel = create_voice_channel
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        result = await handler(
            {"server_id": "1", "name": "ops", "bitrate": 64000},
            {"gateway": gateway},
        )

        self.assertIn("Created voice channel", result[0].text)

    async def test_create_forum_channel_uses_forum_creator(self):
        handler = router.TOOL_ROUTER["create_forum_channel"]
        guild = self._guild()

        async def create_forum(**kwargs):
            payload = dict(kwargs)
            payload.pop("name", None)
            return self._channel(81, kwargs["name"], type="forum", **payload)

        guild.create_forum = create_forum
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        result = await handler(
            {"server_id": "1", "name": "support"},
            {"gateway": gateway},
        )

        self.assertIn("Created forum channel", result[0].text)

    async def test_create_text_channel_forwards_new_kwargs(self):
        handler = router.TOOL_ROUTER["create_text_channel"]
        guild = self._guild()
        calls = []

        async def create_text_channel(**kwargs):
            calls.append(kwargs)
            return self._channel(82, kwargs["name"], type="text")

        guild.create_text_channel = create_text_channel
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        result = await handler(
            {
                "server_id": "1",
                "name": "announcements",
                "position": 3,
                "nsfw": True,
                "slowmode_delay": 30,
                "default_auto_archive_duration": 1440,
                "default_thread_slowmode_delay": 10,
                "reason": "launch",
            },
            {"gateway": gateway},
        )

        self.assertIn("Created text channel", result[0].text)
        kwargs = calls[-1]
        self.assertEqual(kwargs["position"], 3)
        self.assertIs(kwargs["nsfw"], True)
        self.assertEqual(kwargs["slowmode_delay"], 30)
        self.assertEqual(kwargs["default_auto_archive_duration"], 1440)
        self.assertEqual(kwargs["default_thread_slowmode_delay"], 10)
        self.assertEqual(kwargs["reason"], "launch")

    async def test_create_forum_channel_forwards_new_kwargs(self):
        handler = router.TOOL_ROUTER["create_forum_channel"]
        guild = self._guild()
        calls = []

        async def create_forum(**kwargs):
            calls.append(kwargs)
            return self._channel(83, kwargs["name"], type="forum")

        guild.create_forum = create_forum
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        result = await handler(
            {
                "server_id": "1",
                "name": "support",
                "position": 2,
                "default_layout": 2,
                "default_sort_order": 1,
                "default_thread_slowmode_delay": 5,
                "reason": "launch",
            },
            {"gateway": gateway},
        )

        self.assertIn("Created forum channel", result[0].text)
        kwargs = calls[-1]
        self.assertEqual(kwargs["position"], 2)
        self.assertIs(kwargs["default_layout"], discord.ForumLayoutType.gallery_view)
        self.assertIs(kwargs["default_sort_order"], discord.ForumOrderType.creation_date)
        self.assertEqual(kwargs["default_thread_slowmode_delay"], 5)
        self.assertEqual(kwargs["reason"], "launch")

    async def test_create_forum_channel_rejects_bad_default_layout(self):
        handler = router.TOOL_ROUTER["create_forum_channel"]
        guild = self._guild()
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        with self.assertRaisesRegex(ValueError, "default_layout must be one of"):
            await handler(
                {"server_id": "1", "name": "support", "default_layout": 9},
                {"gateway": gateway},
            )

    async def test_update_forum_channel_forwards_layout_and_sort_order(self):
        handler = router.TOOL_ROUTER["update_forum_channel"]
        channel = self._channel(90, "forum", type="forum", available_tags=[])
        guild = self._guild(channels=[channel])
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        result = await handler(
            {
                "server_id": "1",
                "channel_id": "90",
                "default_layout": 1,
                "default_sort_order": 0,
                "default_thread_slowmode_delay": 7,
            },
            {"gateway": gateway},
        )

        self.assertIn("Updated forum channel", result[0].text)
        edit = channel.edit_calls[-1]
        self.assertIs(edit["default_layout"], discord.ForumLayoutType.list_view)
        self.assertIs(edit["default_sort_order"], discord.ForumOrderType.latest_activity)
        self.assertEqual(edit["default_thread_slowmode_delay"], 7)

    async def test_delete_channel_dry_run_default_and_reason_sensitive_token(self):
        handler = router.TOOL_ROUTER["delete_channel"]
        channel = self._channel(70, "general", type="text")
        gateway = SimpleNamespace(fetch_channel=self._async_value(channel))

        first = await handler(
            {"channel_id": "70", "reason": "cleanup"}, {"gateway": gateway}
        )
        second = await handler(
            {"channel_id": "70", "reason": "duplicate"}, {"gateway": gateway}
        )

        first_payload = json.loads(first[0].text)
        second_payload = json.loads(second[0].text)
        self.assertEqual(first_payload["status"], "dry_run")
        self.assertEqual(first_payload["action"], "delete_channel")
        self.assertNotEqual(first_payload["confirmToken"], second_payload["confirmToken"])
        self.assertFalse(hasattr(channel, "deleted"))

    async def test_delete_channel_requires_confirm_token(self):
        handler = router.TOOL_ROUTER["delete_channel"]
        channel = self._channel(71, "general", type="text")
        gateway = SimpleNamespace(fetch_channel=self._async_value(channel))

        with self.assertRaisesRegex(ValueError, "confirm_token"):
            await handler(
                {"channel_id": "71", "reason": "cleanup", "dry_run": False},
                {"gateway": gateway},
            )
        self.assertFalse(hasattr(channel, "deleted"))

    async def test_delete_channel_executes_with_valid_token(self):
        handler = router.TOOL_ROUTER["delete_channel"]
        channel = self._channel(72, "general", type="text")
        gateway = SimpleNamespace(fetch_channel=self._async_value(channel))
        arguments = {"channel_id": "72", "reason": "cleanup"}

        dry = await handler(arguments, {"gateway": gateway})
        token = json.loads(dry[0].text)["confirmToken"]
        result = await handler(
            {**arguments, "dry_run": False, "confirm_token": token},
            {"gateway": gateway},
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "executed")
        self.assertTrue(channel.deleted)

    async def test_delete_channel_requires_reason(self):
        handler = router.TOOL_ROUTER["delete_channel"]
        gateway = SimpleNamespace(fetch_channel=self._async_value(object()))

        with self.assertRaisesRegex(ValueError, "reason is required"):
            await handler({"channel_id": "73"}, {"gateway": gateway})

    def test_schemas_declare_the_new_channel_parameters(self):
        by_name = {tool.name: tool for tool in schemas.compose_tool_registry()}
        expected = {
            "create_text_channel": [
                "position",
                "nsfw",
                "slowmode_delay",
                "default_auto_archive_duration",
                "default_thread_slowmode_delay",
                "reason",
            ],
            "create_voice_channel": ["reason"],
            "create_forum_channel": [
                "reason",
                "default_sort_order",
                "position",
                "default_layout",
                "default_thread_slowmode_delay",
            ],
            "update_forum_channel": [
                "default_layout",
                "default_sort_order",
                "default_thread_slowmode_delay",
            ],
        }
        for tool_name, fields in expected.items():
            schema = by_name[tool_name].input_schema
            properties = schema["properties"]
            for field in fields:
                with self.subTest(tool=tool_name, field=field):
                    self.assertIn(field, properties)
        delete_schema = by_name["delete_channel"].input_schema
        self.assertIn("dry_run", delete_schema["properties"])
        self.assertIn("confirm_token", delete_schema["properties"])
        self.assertIn("reason", delete_schema["required"])

    async def test_read_tools_expose_admin_workflow_fields(self):
        guild = self._guild(
            channels=[
                self._channel(10, "General", type="category", position=0),
                self._channel(
                    70,
                    "general",
                    type="text",
                    position=1,
                    category_id=10,
                    topic="hello",
                    nsfw=True,
                    bitrate=96000,
                    user_limit=0,
                    available_tags=[{"name": "help"}],
                ),
            ]
        )
        gateway = SimpleNamespace(
            resolve_guild=self._async_value(guild),
            fetch_channel=self._async_value(guild.channels[0]),
        )

        structured = await handle_get_channels_structured(
            {"server_id": "1"}, {"gateway": gateway}
        )
        hierarchy = await handle_get_channel_hierarchy(
            {"server_id": "1"}, {"gateway": gateway}
        )
        overwrites = await handle_get_permission_overwrites(
            {"channel_id": "70"}, {"gateway": gateway}
        )

        structured_payload = json.loads(structured[0].text)
        hierarchy_payload = json.loads(hierarchy[0].text)
        overwrites_payload = json.loads(overwrites[0].text)

        channel_payload = next(
            item for item in structured_payload["channels"] if item["name"] == "general"
        )
        self.assertEqual(channel_payload["topic"], "hello")
        self.assertIn("children", hierarchy_payload["categories"][0])
        self.assertIn("overwrites", overwrites_payload)

    @staticmethod
    def _channel(channel_id, name, **attrs):
        data = {"id": channel_id, "name": name, "edit_calls": [], "overwrites": {}}

        async def edit(**kwargs):
            data["edit_calls"].append(kwargs)
            for key, value in kwargs.items():
                setattr(channel, key, value)

        async def delete(**_kwargs):
            channel.deleted = True

        data["edit"] = edit
        data["delete"] = delete
        data.update(attrs)
        channel = SimpleNamespace(**data)
        return channel

    @staticmethod
    def _guild(**attrs):
        defaults = {"id": 1, "channels": []}
        defaults.update(attrs)
        return SimpleNamespace(**defaults)

    @staticmethod
    def _async_value(value):
        async def inner(*_args, **_kwargs):
            return value

        return inner


class ForumTagIdPreservationTests(unittest.TestCase):
    """Regression: rewriting forum tags without ids orphaned every tagged post.

    Discord keys applied tags by tag id, and ``ForumTag.to_dict`` only emits
    ``id`` when the instance already carries one while ``ForumTag.__init__``
    takes no ``id`` argument. A handler that builds tags without assigning the
    id therefore asks Discord to recreate every tag, and each forum post that
    had one loses it.
    """

    @staticmethod
    def _guild():
        return SimpleNamespace(emojis=[])

    def test_existing_tag_id_is_sent_to_discord(self):
        tags = _build_forum_tags(
            self._guild(),
            [{"id": "1443173616320512060", "name": "Nhật Ký", "emoji": "📜",
              "emojiId": None, "emojiAnimated": False, "moderated": False}],
            "available_tags",
        )
        self.assertEqual(tags[0].to_dict()["id"], 1443173616320512060)

    def test_read_payload_round_trips_the_id(self):
        """get_channels_structured emits id/emojiId; feeding it back must keep both."""
        tag = SimpleNamespace(
            id=1443173616320512060, name="Suy Ngẫm",
            emoji=SimpleNamespace(name="☁️", id=None, animated=False),
            moderated=False,
        )
        payload = _serialize_forum_tag(tag)
        rebuilt = _build_forum_tags(self._guild(), [payload], "available_tags")
        self.assertEqual(rebuilt[0].to_dict()["id"], 1443173616320512060)
        self.assertEqual(rebuilt[0].name, "Suy Ngẫm")

    def test_tag_id_is_never_used_as_the_emoji_id(self):
        """Regression: the tag's own id leaked into parse_emoji as an emoji id.

        parse_emoji falls back to ``id`` when ``emojiId`` is absent, so passing
        the whole tag object produced ``<:💡:<tag id>`` and Discord answered
        "Invalid emoji id or name".
        """
        tags = _build_forum_tags(
            self._guild(),
            [{"id": "1443173616320512060", "name": "Tips", "emoji": "💡"}],
            "available_tags",
        )
        payload = tags[0].to_dict()
        self.assertEqual(payload["emoji_id"], None)
        self.assertEqual(payload["emoji_name"], "💡")
        self.assertNotIn(str(1443173616320512060), str(payload["emoji_name"]))

    def test_custom_emoji_id_is_still_honoured(self):
        tags = _build_forum_tags(
            self._guild(),
            [{"id": "1443173616320512060", "name": "X", "emoji": "custom",
              "emojiId": "999888777", "emojiAnimated": False}],
            "available_tags",
        )
        self.assertIn("999888777", tags[0].to_dict()["emoji_name"])

    def test_a_brand_new_tag_sends_no_id(self):
        tags = _build_forum_tags(self._guild(), [{"name": "New", "emoji": "🎉"}], "available_tags")
        self.assertNotIn("id", tags[0].to_dict())

    def test_guard_blocks_a_rewrite_that_would_orphan_tags(self):
        channel = SimpleNamespace(available_tags=[
            SimpleNamespace(id=1, name="Nhật Ký"), SimpleNamespace(id=2, name="Tư duy"),
        ])
        incoming = _build_forum_tags(
            self._guild(), [{"name": "Nhật Ký", "emoji": "📜"}], "available_tags"
        )
        with self.assertRaisesRegex(ValueError, "would drop existing tag"):
            _guard_tag_ids(channel, incoming, "available_tags", allow_recreate=False)

    def test_guard_allows_a_rename_that_keeps_every_id(self):
        channel = SimpleNamespace(available_tags=[
            SimpleNamespace(id=1, name="Nhật Ký"), SimpleNamespace(id=2, name="Tư duy"),
        ])
        incoming = _build_forum_tags(
            self._guild(),
            [{"id": "1", "name": "Nhật Ký 2"}, {"id": "2", "name": "Tư duy 2"}],
            "available_tags",
        )
        _guard_tag_ids(channel, incoming, "available_tags", allow_recreate=False)

    def test_guard_can_be_overridden_explicitly(self):
        channel = SimpleNamespace(available_tags=[SimpleNamespace(id=1, name="Nhật Ký")])
        incoming = _build_forum_tags(self._guild(), [{"name": "Tư duy"}], "available_tags")
        _guard_tag_ids(channel, incoming, "available_tags", allow_recreate=True)


if __name__ == "__main__":
    unittest.main()
