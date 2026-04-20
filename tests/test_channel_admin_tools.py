import json
import os
import sys
import unittest
import types
from types import SimpleNamespace


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

sys.modules.setdefault(
    "discord",
    types.ModuleType("discord"),
)
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

mcp = types.ModuleType("mcp")
mcp_server = types.ModuleType("mcp.server")


class _Server:
    def __init__(self, *args, **kwargs):
        pass

    def list_tools(self):
        def decorator(func):
            return func

        return decorator

    def call_tool(self):
        def decorator(func):
            return func

        return decorator

    async def run(self, *args, **kwargs):
        return None

    def create_initialization_options(self):
        return None


mcp_server.Server = _Server
mcp_server_stdio = types.ModuleType("mcp.server.stdio")
mcp_server_stdio.stdio_server = lambda: None
mcp_types = types.ModuleType("mcp.types")


class _Tool:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _TextContent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


mcp_types.TextContent = _TextContent
mcp_types.Tool = _Tool
sys.modules.setdefault("mcp", mcp)
sys.modules.setdefault("mcp.server", mcp_server)
sys.modules.setdefault("mcp.server.stdio", mcp_server_stdio)
sys.modules.setdefault("mcp.types", mcp_types)

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

from discord_mcp.tools.handlers.inventory import (
    handle_get_channel_hierarchy,
    handle_get_channels_structured,
    handle_get_permission_overwrites,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER
from discord_mcp.tools.schemas import compose_tool_registry


class ChannelAdminToolRegistryTests(unittest.TestCase):
    def test_channel_admin_tools_are_present_in_schema_registry(self):
        names = [tool.name for tool in compose_tool_registry()]

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
            self.assertIn(tool_name, TOOL_ROUTER)


class ChannelAdminHandlerContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_text_channel_maps_each_field(self):
        handler = TOOL_ROUTER["update_text_channel"]
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

    async def test_update_voice_channel_maps_each_field(self):
        handler = TOOL_ROUTER["update_voice_channel"]
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

    async def test_update_forum_channel_maps_each_field(self):
        handler = TOOL_ROUTER["update_forum_channel"]
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

    async def test_update_text_channel_rejects_wrong_channel_type(self):
        handler = TOOL_ROUTER["update_text_channel"]
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
        handler = TOOL_ROUTER["update_voice_channel"]
        guild = self._guild(channels=[])
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        with self.assertRaisesRegex((KeyError, ValueError), "channel_id"):
            await handler({"server_id": "1", "name": "voice"}, {"gateway": gateway})

    async def test_update_forum_channel_rejects_unknown_fields(self):
        handler = TOOL_ROUTER["update_forum_channel"]
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

    async def test_update_forum_channel_rejects_library_unsupported_fields(self):
        handler = TOOL_ROUTER["update_forum_channel"]
        guild = self._guild(
            channels=[
                self._channel(60, "forum", type="forum", available_tags=[]),
            ]
        )
        gateway = SimpleNamespace(resolve_guild=self._async_value(guild))

        with self.assertRaisesRegex(ValueError, "field_not_supported_by_library"):
            await handler(
                {
                    "server_id": "1",
                    "channel_id": "60",
                    "name": "forum",
                    "default_sort_order": 1,
                },
                {"gateway": gateway},
            )

    async def test_read_tools_expose_admin_workflow_fields(self):
        guild = self._guild(
            channels=[
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
                )
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

        self.assertIn("nsfw", structured_payload["channels"][0])
        self.assertIn("bitrate", structured_payload["channels"][0])
        self.assertIn("availableTags", structured_payload["channels"][0])
        self.assertIn("children", hierarchy_payload["categories"][0])
        self.assertIn("overwrites", overwrites_payload)

    @staticmethod
    def _channel(channel_id, name, **attrs):
        data = {"id": channel_id, "name": name}
        data.update(attrs)
        return SimpleNamespace(**data)

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
