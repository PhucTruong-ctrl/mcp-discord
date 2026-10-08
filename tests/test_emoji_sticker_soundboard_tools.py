import json
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

import discord  # noqa: E402

from discord_mcp.tools.handlers import emoji_sticker_soundboard as tools  # noqa: E402
from discord_mcp.tools.handlers.emoji_sticker_soundboard import (  # noqa: E402
    handle_create_application_emoji,
    handle_create_emoji,
    handle_create_soundboard_sound,
    handle_create_sticker,
    handle_delete_application_emoji,
    handle_delete_emoji,
    handle_delete_soundboard_sound,
    handle_delete_sticker,
    handle_edit_application_emoji,
    handle_edit_emoji,
    handle_edit_soundboard_sound,
    handle_edit_sticker,
    handle_list_application_emojis,
    handle_list_soundboard_sounds,
    handle_list_stickers,
    handle_send_soundboard_sound,
)
from discord_mcp.tools.schemas.emoji_sticker_soundboard import (  # noqa: E402
    EMOJI_STICKER_SOUNDBOARD_TOOLS,
)

EXPECTED_TOOL_NAMES = [
    "create_emoji",
    "edit_emoji",
    "delete_emoji",
    "create_application_emoji",
    "edit_application_emoji",
    "delete_application_emoji",
    "list_application_emojis",
    "create_sticker",
    "edit_sticker",
    "delete_sticker",
    "list_stickers",
    "create_soundboard_sound",
    "list_soundboard_sounds",
    "edit_soundboard_sound",
    "delete_soundboard_sound",
    "send_soundboard_sound",
]
GATED_TOOLS = {
    "create_emoji",
    "edit_emoji",
    "delete_emoji",
    "create_application_emoji",
    "edit_application_emoji",
    "delete_application_emoji",
    "create_sticker",
    "edit_sticker",
    "delete_sticker",
    "create_soundboard_sound",
    "edit_soundboard_sound",
    "delete_soundboard_sound",
}
UNGATED_TOOLS = {
    "list_application_emojis",
    "list_stickers",
    "list_soundboard_sounds",
    "send_soundboard_sound",
}
REASON_REQUIRED_TOOLS = {
    "create_sticker",
    "delete_emoji",
    "delete_sticker",
    "delete_soundboard_sound",
}
READ_ONLY_TOOLS = {
    "list_application_emojis",
    "list_stickers",
    "list_soundboard_sounds",
}

STICKER_ROW_KEYS = {
    "id",
    "name",
    "description",
    "emoji",
    "format",
    "available",
    "tags",
    "guildId",
    "userId",
}
SOUND_ROW_KEYS = {"id", "name", "volume", "emoji", "available", "userId", "url"}


class FakeRole:
    def __init__(self, role_id, name="mods"):
        self.id = int(role_id)
        self.name = name


class FakeEmoji:
    def __init__(self, emoji_id, name="wave", guild_id=9001):
        self.id = int(emoji_id)
        self.name = name
        self.guild_id = guild_id
        self.roles = []
        self.animated = False
        self.available = True
        self.managed = False
        self.require_colons = True
        self.url = f"https://cdn.example/emojis/{emoji_id}.png"
        self.edit_kwargs = None

    async def edit(self, **kwargs):
        self.edit_kwargs = dict(kwargs)
        if "name" in kwargs:
            self.name = kwargs["name"]
        return self

    async def delete(self, *, reason=None):
        raise AssertionError("delete_emoji must go through Guild.delete_emoji")


class FakeSticker:
    """Deletion goes through Guild.delete_sticker; edit mirrors GuildSticker.edit."""

    def __init__(self, sticker_id, name="airhorn-sticker"):
        self.id = int(sticker_id)
        self.name = name
        self.description = "sticker description"
        self.emoji = "\U0001f600"
        self.format = discord.StickerFormatType.png
        self.available = True
        self.tags = "cat, cute"
        self.guild_id = 9001
        self.user = None
        self.edit_kwargs = None

    async def edit(self, **kwargs):
        self.edit_kwargs = dict(kwargs)
        if "name" in kwargs:
            self.name = kwargs["name"]
        if "description" in kwargs:
            self.description = kwargs["description"]
        if "emoji" in kwargs:
            self.emoji = kwargs["emoji"]
        return self


class FakeSound:
    def __init__(self, sound_id, name="airhorn", guild=None):
        self.id = int(sound_id)
        self.name = name
        self.volume = 1.0
        self.emoji = None
        self.available = True
        self.user = None
        self.url = f"https://cdn.example/sounds/{sound_id}.mp3"
        self.guild = guild
        self.edit_kwargs = None
        self.deleted_with = []

    async def edit(self, **kwargs):
        self.edit_kwargs = dict(kwargs)
        if "name" in kwargs:
            self.name = kwargs["name"]
        if "volume" in kwargs:
            self.volume = kwargs["volume"]
        return self

    async def delete(self, *, reason=None):
        self.deleted_with.append(reason)


class FakeVoiceChannel:
    def __init__(self, channel_id, guild, name="Lobby"):
        self.id = int(channel_id)
        self.name = name
        self.type = discord.ChannelType.voice
        self.guild = guild
        self.sent = []

    async def send_sound(self, sound):
        self.sent.append(sound)


class FakeTextChannel:
    def __init__(self, channel_id, guild, name="general"):
        self.id = int(channel_id)
        self.name = name
        self.type = discord.ChannelType.text
        self.guild = guild


class FakeGuild:
    def __init__(self, emojis=(), roles=(), stickers=(), sounds=(), channels=()):
        self.id = 9001
        self.name = "TestServer"
        self.emojis = list(emojis)
        self.roles = list(roles)
        self.stickers = list(stickers)
        self.channels = list(channels)
        self._sounds = {sound.id: sound for sound in sounds}
        self.created_emoji = None
        self.deleted_emoji = []
        self.created_sticker = None
        self.deleted_sticker = []
        self.created_sound = None

    def get_emoji(self, emoji_id):
        return next((e for e in self.emojis if e.id == emoji_id), None)

    def get_role(self, role_id):
        return next((r for r in self.roles if r.id == role_id), None)

    def get_channel(self, channel_id):
        return next((c for c in self.channels if c.id == channel_id), None)

    def get_soundboard_sound(self, sound_id):
        return self._sounds.get(sound_id)

    async def fetch_stickers(self):
        return list(self.stickers)

    async def fetch_sticker(self, sticker_id):
        sticker = next(
            (s for s in self.stickers if s.id == sticker_id), None
        )
        if sticker is None:
            raise discord.NotFound(
                _NotFoundResponse(), {"message": "Unknown Sticker", "code": 10013}
            )
        return sticker

    async def fetch_soundboard_sounds(self):
        return list(self._sounds.values())

    async def fetch_channel(self, channel_id):
        return self.get_channel(channel_id)

    async def create_custom_emoji(self, **kwargs):
        self.created_emoji = dict(kwargs)
        return FakeEmoji(555000, name=kwargs["name"], guild_id=self.id)

    async def delete_emoji(self, emoji, *, reason=None):
        self.deleted_emoji.append((emoji.id, reason))

    async def create_sticker(self, **kwargs):
        self.created_sticker = dict(kwargs)
        return FakeSticker(777000, name=kwargs["name"])

    async def delete_sticker(self, sticker, *, reason=None):
        self.deleted_sticker.append((sticker.id, reason))

    async def create_soundboard_sound(self, **kwargs):
        self.created_sound = dict(kwargs)
        return FakeSound(888000, name=kwargs["name"], guild=self)


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, server_id):
        return self.guild


class _NotFoundResponse:
    status = 404
    reason = "Not Found"


class FakeApplicationHttp:
    def __init__(self):
        self.deleted = []

    async def delete_application_emoji(self, application_id, emoji_id):
        self.deleted.append((application_id, emoji_id))


class FakeClient:
    def __init__(self, app_emojis=()):
        self.application_id = 777000
        self.http = FakeApplicationHttp()
        self.app_emojis = {emoji.id: emoji for emoji in app_emojis}
        self.created = None

    async def create_application_emoji(self, *, name, image):
        self.created = {"name": name, "image": image}
        return FakeEmoji(900100, name=name, guild_id=0)

    async def fetch_application_emoji(self, emoji_id):
        emoji = self.app_emojis.get(emoji_id)
        if emoji is None:
            raise discord.NotFound(
                _NotFoundResponse(), {"message": "Unknown Emoji", "code": 10014}
            )
        return emoji

    async def fetch_application_emojis(self):
        return list(self.app_emojis.values())


class EmojiStickerSoundboardSchemaTests(unittest.TestCase):
    def test_sixteen_tools_matching_the_contract(self):
        self.assertEqual(len(EMOJI_STICKER_SOUNDBOARD_TOOLS), 16)
        self.assertEqual(
            [tool.name for tool in EMOJI_STICKER_SOUNDBOARD_TOOLS],
            EXPECTED_TOOL_NAMES,
        )

    def test_gate_properties_only_on_gated_tools(self):
        for tool in EMOJI_STICKER_SOUNDBOARD_TOOLS:
            properties = tool.input_schema["properties"]
            if tool.name in GATED_TOOLS:
                self.assertIn("dry_run", properties, tool.name)
                self.assertIn("confirm_token", properties, tool.name)
                self.assertTrue(
                    properties["dry_run"].get("default", None) is True, tool.name
                )
            else:
                self.assertNotIn("dry_run", properties, tool.name)
                self.assertNotIn("confirm_token", properties, tool.name)
        self.assertEqual(
            GATED_TOOLS | UNGATED_TOOLS, set(EXPECTED_TOOL_NAMES)
        )

    def test_reason_required_where_the_table_requires_it(self):
        for tool in EMOJI_STICKER_SOUNDBOARD_TOOLS:
            required = tool.input_schema["required"]
            if tool.name in REASON_REQUIRED_TOOLS:
                self.assertIn("reason", required, tool.name)
            else:
                self.assertNotIn("reason", required, tool.name)

    def test_read_only_tools_take_no_gate_or_reason(self):
        for tool in EMOJI_STICKER_SOUNDBOARD_TOOLS:
            if tool.name in READ_ONLY_TOOLS:
                required = tool.input_schema["required"]
                properties = tool.input_schema["properties"]
                for key in ("reason", "dry_run", "confirm_token"):
                    self.assertNotIn(key, properties, tool.name)
                    self.assertNotIn(key, required, tool.name)


class EmojiStickerSoundboardHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.role = FakeRole(42)
        self.emoji = FakeEmoji(111, name="party")
        self.sticker = FakeSticker(222)
        self.text_channel = FakeTextChannel(333, guild=None)
        self.voice_channel = FakeVoiceChannel(444, guild=None)
        self.sound = FakeSound(555, guild=None)
        self.guild = FakeGuild(
            emojis=[self.emoji],
            roles=[self.role],
            stickers=[self.sticker],
            sounds=[self.sound],
            channels=[self.text_channel, self.voice_channel],
        )
        self.text_channel.guild = self.guild
        self.voice_channel.guild = self.guild
        self.sound.guild = self.guild
        self.gateway = FakeGateway(self.guild)
        self.app_emoji = FakeEmoji(666, name="app-wave", guild_id=0)
        self.client = FakeClient(app_emojis=[self.app_emoji])
        self.deps = {"gateway": self.gateway, "discord_client": self.client}

    async def _call(self, handler, arguments, deps=None):
        result = await handler(arguments, self.deps if deps is None else deps)
        return json.loads(result[0].text)

    async def _execute(self, handler, arguments):
        dry = await self._call(handler, {**arguments, "dry_run": True})
        token = dry["confirmToken"]
        return await self._call(
            handler, {**arguments, "dry_run": False, "confirm_token": token}
        )

    async def test_every_handler_requires_gateway(self):
        cases = [
            (handle_create_emoji, {"server_id": "9001", "name": "x", "image_url": "https://example.com/x.png"}),
            (handle_edit_emoji, {"server_id": "9001", "emoji_id": "111", "name": "y"}),
            (handle_delete_emoji, {"server_id": "9001", "emoji_id": "111", "reason": "cleanup"}),
            (handle_create_application_emoji, {"name": "x", "image_url": "https://example.com/x.png"}),
            (handle_edit_application_emoji, {"emoji_id": "666", "name": "y"}),
            (handle_delete_application_emoji, {"emoji_id": "666"}),
            (handle_list_application_emojis, {}),
            (handle_create_sticker, {"server_id": "9001", "name": "x", "description": "d", "emoji": "\U0001f600", "file_path": "x.png", "reason": "r"}),
            (handle_edit_sticker, {"server_id": "9001", "sticker_id": "222", "name": "y"}),
            (handle_delete_sticker, {"server_id": "9001", "sticker_id": "222", "reason": "cleanup"}),
            (handle_list_stickers, {"server_id": "9001"}),
            (handle_create_soundboard_sound, {"server_id": "9001", "name": "x", "sound_url": "https://example.com/x.mp3"}),
            (handle_list_soundboard_sounds, {"server_id": "9001"}),
            (handle_edit_soundboard_sound, {"server_id": "9001", "sound_id": "555", "name": "y"}),
            (handle_delete_soundboard_sound, {"server_id": "9001", "sound_id": "555", "reason": "cleanup"}),
            (handle_send_soundboard_sound, {"server_id": "9001", "channel_id": "444", "sound_id": "555"}),
        ]
        self.assertEqual(len(cases), 16)
        for handler, arguments in cases:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "gateway is required"):
                    await handler(arguments, {})

    async def test_create_emoji_dry_run_returns_confirm_token(self):
        with patch("discord_mcp.core.common.download_url", return_value=b"image-bytes"):
            payload = await self._call(
                handle_create_emoji,
                {
                    "server_id": "9001",
                    "name": "party",
                    "image_url": "https://example.com/party.png",
                    "role_ids": ["42"],
                },
            )
        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["action"], "create_emoji")
        self.assertTrue(payload["confirmToken"])
        self.assertIsNone(self.guild.created_emoji)

    async def test_create_emoji_execute_without_token_raises(self):
        with patch("discord_mcp.core.common.download_url", return_value=b"image-bytes"):
            with self.assertRaisesRegex(ValueError, "confirm_token is required"):
                await self._call(
                    handle_create_emoji,
                    {
                        "server_id": "9001",
                        "name": "party",
                        "image_url": "https://example.com/party.png",
                        "dry_run": False,
                    },
                )
        self.assertIsNone(self.guild.created_emoji)

    async def test_create_emoji_executes_with_token(self):
        with patch("discord_mcp.core.common.download_url", return_value=b"image-bytes"):
            payload = await self._execute(
                handle_create_emoji,
                {
                    "server_id": "9001",
                    "name": "party",
                    "image_url": "https://example.com/party.png",
                    "role_ids": ["42"],
                    "reason": "event prep",
                },
            )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "create_emoji")
        self.assertEqual(payload["emoji"]["name"], "party")
        created = self.guild.created_emoji
        self.assertEqual(created["name"], "party")
        self.assertEqual(created["image"], b"image-bytes")
        self.assertEqual(created["roles"], [self.role])
        self.assertEqual(created["reason"], "event prep")

    async def test_create_emoji_rejects_non_http_url_before_gate(self):
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_create_emoji,
                {
                    "server_id": "9001",
                    "name": "party",
                    "image_url": "file:///etc/passwd",
                },
            )
        message = str(ctx.exception)
        self.assertIn("only http/https", message)
        self.assertIn("file:///etc/passwd", message)
        self.assertIsNone(self.guild.created_emoji)

    async def test_delete_emoji_requires_reason_before_gate(self):
        with self.assertRaisesRegex(ValueError, "reason is required for delete_emoji"):
            await self._call(
                handle_delete_emoji, {"server_id": "9001", "emoji_id": "111"}
            )
        self.assertEqual(self.guild.deleted_emoji, [])

    async def test_delete_emoji_gate_and_execute_payload_keys(self):
        payload = await self._call(
            handle_delete_emoji,
            {"server_id": "9001", "emoji_id": "111", "reason": "cleanup"},
        )
        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(self.guild.deleted_emoji, [])

        executed = await self._execute(
            handle_delete_emoji,
            {"server_id": "9001", "emoji_id": "111", "reason": "cleanup"},
        )
        self.assertEqual(executed["status"], "executed")
        self.assertEqual(executed["action"], "delete_emoji")
        self.assertEqual(executed["serverId"], "9001")
        self.assertEqual(executed["emojiId"], "111")
        self.assertEqual(self.guild.deleted_emoji, [(111, "cleanup")])

    async def test_delete_emoji_rejects_unknown_emoji(self):
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_delete_emoji,
                {"server_id": "9001", "emoji_id": "999", "reason": "cleanup"},
            )
        message = str(ctx.exception)
        self.assertIn("999", message)
        self.assertIn("TestServer", message)

    async def test_create_sticker_rejects_missing_file(self):
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_create_sticker,
                {
                    "server_id": "9001",
                    "name": "cool-sticker",
                    "description": "desc",
                    "emoji": "\U0001f600",
                    "file_path": "/nonexistent/sticker.png",
                    "reason": "prep",
                },
            )
        message = str(ctx.exception)
        self.assertIn("/nonexistent/sticker.png", message)
        self.assertIn("not an existing file", message)
        self.assertIsNone(self.guild.created_sticker)

    async def test_list_stickers_payload_keys(self):
        payload = await self._call(handle_list_stickers, {"server_id": "9001"})
        self.assertEqual(payload["serverId"], "9001")
        self.assertEqual(payload["count"], 1)
        row = payload["stickers"][0]
        self.assertTrue(STICKER_ROW_KEYS.issubset(row), set(row))
        self.assertEqual(row["id"], "222")
        self.assertEqual(row["guildId"], "9001")
        self.assertNotIn("sortedId", row)

    async def test_edit_sticker_dry_run_returns_confirm_token(self):
        payload = await self._call(
            handle_edit_sticker,
            {"server_id": "9001", "sticker_id": "222", "name": "new-sticker"},
        )
        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["action"], "edit_sticker")
        self.assertTrue(payload["confirmToken"])
        self.assertIsNone(self.sticker.edit_kwargs)

    async def test_edit_sticker_dry_run_details_list_only_supplied_fields(self):
        payload = await self._call(
            handle_edit_sticker,
            {"server_id": "9001", "sticker_id": "222", "name": "new-sticker"},
        )
        self.assertEqual(payload["details"]["fields"], {"name": "new-sticker"})

    async def test_edit_sticker_execute_without_token_raises(self):
        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await self._call(
                handle_edit_sticker,
                {
                    "server_id": "9001",
                    "sticker_id": "222",
                    "name": "new-sticker",
                    "dry_run": False,
                },
            )
        self.assertIsNone(self.sticker.edit_kwargs)

    async def test_edit_sticker_execute_with_token_payload_keys(self):
        payload = await self._execute(
            handle_edit_sticker,
            {
                "server_id": "9001",
                "sticker_id": "222",
                "name": "new-sticker",
                "description": "new description",
                "emoji": "\U0001f389",
                "reason": "retag",
            },
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "edit_sticker")
        row = payload["sticker"]
        self.assertTrue(STICKER_ROW_KEYS.issubset(row), set(row))
        self.assertEqual(row["id"], "222")
        self.assertEqual(
            self.sticker.edit_kwargs,
            {
                "name": "new-sticker",
                "description": "new description",
                "emoji": "\U0001f389",
                "reason": "retag",
            },
        )

    async def test_edit_sticker_rejects_missing_sticker(self):
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_edit_sticker,
                {"server_id": "9001", "sticker_id": "999", "name": "new-sticker"},
            )
        message = str(ctx.exception)
        self.assertIn("999", message)
        self.assertIn("TestServer", message)

    async def test_list_application_emojis_payload_keys(self):
        payload = await self._call(handle_list_application_emojis, {})
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["emojis"][0]["id"], "666")
        self.assertEqual(payload["emojis"][0]["name"], "app-wave")

    async def test_create_application_emoji_gate_round_trip(self):
        with patch("discord_mcp.core.common.download_url", return_value=b"app-bytes"):
            payload = await self._execute(
                handle_create_application_emoji,
                {"name": "app-wave", "image_url": "https://example.com/a.png"},
            )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "create_application_emoji")
        self.assertEqual(self.client.created, {"name": "app-wave", "image": b"app-bytes"})
        self.assertEqual(payload["emoji"]["name"], "app-wave")

    async def test_delete_application_emoji_uses_http_delete_path(self):
        payload = await self._execute(
            handle_delete_application_emoji, {"emoji_id": "666"}
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "delete_application_emoji")
        self.assertEqual(payload["emojiId"], "666")
        self.assertEqual(self.client.http.deleted, [(777000, 666)])

    async def test_delete_application_emoji_rejects_unknown_id(self):
        with self.assertRaises(ValueError) as ctx:
            await self._call(handle_delete_application_emoji, {"emoji_id": "404404"})
        self.assertIn("404404", str(ctx.exception))
        self.assertEqual(self.client.http.deleted, [])

    async def test_create_soundboard_sound_rejects_out_of_range_volume(self):
        with self.assertRaisesRegex(
            ValueError, "volume must be between 0.0 and 1.0"
        ):
            await self._call(
                handle_create_soundboard_sound,
                {
                    "server_id": "9001",
                    "name": "horn",
                    "sound_url": "https://example.com/horn.mp3",
                    "volume": 1.5,
                },
            )
        self.assertIsNone(self.guild.created_sound)

    async def test_create_soundboard_sound_rejects_non_numeric_volume(self):
        with self.assertRaisesRegex(ValueError, "volume must be a number"):
            await self._call(
                handle_create_soundboard_sound,
                {
                    "server_id": "9001",
                    "name": "horn",
                    "sound_url": "https://example.com/horn.mp3",
                    "volume": "loud",
                },
            )

    async def test_create_soundboard_sound_gate_round_trip(self):
        with patch("discord_mcp.core.common.download_url", return_value=b"sound-bytes"):
            dry = await self._call(
                handle_create_soundboard_sound,
                {
                    "server_id": "9001",
                    "name": "horn",
                    "sound_url": "https://example.com/horn.mp3",
                    "volume": 0.5,
                    "emoji": "<:party:111>",
                },
            )
        self.assertEqual(dry["status"], "dry_run")
        self.assertTrue(dry["confirmToken"])

        with patch("discord_mcp.core.common.download_url", return_value=b"sound-bytes"):
            payload = await self._execute(
                handle_create_soundboard_sound,
                {
                    "server_id": "9001",
                    "name": "horn",
                    "sound_url": "https://example.com/horn.mp3",
                    "volume": 0.5,
                    "emoji": "<:party:111>",
                    "reason": "fun",
                },
            )
        self.assertEqual(payload["status"], "executed")
        created = self.guild.created_sound
        self.assertEqual(created["sound"], b"sound-bytes")
        self.assertEqual(created["volume"], 0.5)
        self.assertEqual(created["emoji"], "<:party:111>")
        self.assertEqual(created["reason"], "fun")
        self.assertEqual(payload["sound"]["name"], "horn")

    async def test_list_soundboard_sounds_payload_keys(self):
        payload = await self._call(
            handle_list_soundboard_sounds, {"server_id": "9001"}
        )
        self.assertEqual(payload["serverId"], "9001")
        self.assertEqual(payload["count"], 1)
        row = payload["sounds"][0]
        self.assertTrue(SOUND_ROW_KEYS.issubset(row), set(row))
        self.assertEqual(row["id"], "555")

    async def test_edit_soundboard_sound_execute_with_token(self):
        payload = await self._execute(
            handle_edit_soundboard_sound,
            {
                "server_id": "9001",
                "sound_id": "555",
                "name": "foghorn",
                "volume": 0.25,
            },
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "edit_soundboard_sound")
        self.assertEqual(
            self.sound.edit_kwargs, {"name": "foghorn", "volume": 0.25}
        )

    async def test_delete_soundboard_sound_execute_without_token_raises(self):
        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await self._call(
                handle_delete_soundboard_sound,
                {
                    "server_id": "9001",
                    "sound_id": "555",
                    "reason": "cleanup",
                    "dry_run": False,
                },
            )
        self.assertEqual(self.sound.deleted_with, [])

    async def test_send_soundboard_sound_rejects_non_voice_channel(self):
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_send_soundboard_sound,
                {"server_id": "9001", "channel_id": "333", "sound_id": "555"},
            )
        message = str(ctx.exception)
        self.assertIn("333", message)
        self.assertIn("text", message)
        self.assertIn("voice channel", message)
        self.assertEqual(self.voice_channel.sent, [])

    async def test_send_soundboard_sound_rejects_foreign_guild_sound(self):
        foreign_guild = FakeGuild()
        foreign_guild.id = 9002
        self.sound.guild = foreign_guild
        with self.assertRaises(ValueError) as ctx:
            await self._call(
                handle_send_soundboard_sound,
                {"server_id": "9001", "channel_id": "444", "sound_id": "555"},
            )
        self.assertIn("different server", str(ctx.exception))
        self.assertEqual(self.voice_channel.sent, [])

    async def test_send_soundboard_sound_plays_in_voice_channel(self):
        payload = await self._call(
            handle_send_soundboard_sound,
            {"server_id": "9001", "channel_id": "444", "sound_id": "555"},
        )
        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["action"], "send_soundboard_sound")
        self.assertEqual(payload["channelId"], "444")
        self.assertEqual(payload["soundId"], "555")
        self.assertEqual(self.voice_channel.sent, [self.sound])


if __name__ == "__main__":
    unittest.main()
