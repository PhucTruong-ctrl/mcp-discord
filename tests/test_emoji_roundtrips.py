"""Read -> write round trips for emoji-bearing payloads (P1-P4 of EMOJI_ROUNDTRIP_ISSUES)."""

import json
import os
import sys
import unittest
from types import SimpleNamespace


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")
os.environ.setdefault(
    "DISCORD_MCP_STATE_DIR",
    os.path.join("/tmp", "discord-mcp-emoji-state"),
)

import discord  # noqa: E402
from discord_mcp.core.emoji import parse_emoji  # noqa: E402
from discord_mcp.core.serialize import (  # noqa: E402
    _serialize_forum_tag,
    _serialize_onboarding,
    _serialize_welcome_screen,
)
from discord_mcp.tools.handlers.channels import (  # noqa: E402
    handle_update_forum_channel,
)
from discord_mcp.tools.handlers.inventory import (  # noqa: E402
    handle_get_channels_structured,
)
from discord_mcp.tools.handlers.onboarding import (  # noqa: E402
    handle_update_guild_onboarding,
    handle_update_guild_welcome_screen,
)

CUSTOM_EMOJI = SimpleNamespace(
    id=1447691293911285780, name="Herta_Kurukuru", animated=True
)


class FakeChannel(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.edit_calls = []

    async def edit(self, **kwargs):
        self.edit_calls.append(kwargs)
        return self


class FakeGuild:
    def __init__(self, channels, onboarding_payload=None):
        self.id = 1
        self.name = "Guild"
        self.channels = list(channels)
        self.default_role = SimpleNamespace(id=1, name="@everyone")
        self._onboarding_payload = onboarding_payload
        self.edit_calls = []

    def get_channel(self, channel_id):
        return next((c for c in self.channels if c.id == channel_id), None)

    def get_role(self, role_id):
        return None

    async def edit_onboarding(self, **kwargs):
        self.edit_calls.append(("onboarding", kwargs))
        return SimpleNamespace(
            enabled=kwargs.get("enabled"),
            mode=None,
            default_channels=[],
            prompts=[],
        )

    async def edit_welcome_screen(self, **kwargs):
        self.edit_calls.append(("welcome_screen", kwargs))
        return SimpleNamespace(
            description=kwargs.get("description"),
            welcome_channels=kwargs.get("welcome_channels") or [],
            enabled=kwargs.get("enabled"),
        )


class FakeGateway:
    def __init__(self, guild, onboarding_payload=None):
        self.guild = guild
        self._onboarding_payload = onboarding_payload

    async def resolve_guild(self, server_id=None):
        return self.guild

    async def fetch_onboarding_payload(self, server_id=None):
        return self._onboarding_payload or {
            "prompts": [],
            "default_channel_ids": [],
            "enabled": True,
            "mode": 0,
        }


def _deps(guild, onboarding_payload=None):
    return {"gateway": FakeGateway(guild, onboarding_payload)}


class ParseEmojiShapeTests(unittest.TestCase):
    """A client that hands a tool's own read payload back must not lose the emoji."""

    ANIMATED = {"emoji": "Donowall", "emojiId": "1447515480079335554", "emojiAnimated": True}
    STATIC = {"emoji": "joecool", "emojiId": "1425049288240529468", "emojiAnimated": False}

    def test_nested_read_shape_resolves(self):
        self.assertEqual(
            parse_emoji(None, {"emoji": self.ANIMATED}),
            "<a:Donowall:1447515480079335554>",
        )
        self.assertEqual(
            parse_emoji(None, {"emoji": self.STATIC}),
            "<:joecool:1425049288240529468>",
        )

    def test_flat_read_shape_still_resolves(self):
        self.assertEqual(parse_emoji(None, self.ANIMATED), "<a:Donowall:1447515480079335554>")
        self.assertEqual(parse_emoji(None, self.STATIC), "<:joecool:1425049288240529468>")

    def test_raw_api_shape_still_resolves(self):
        self.assertEqual(
            parse_emoji(
                None, {"emoji": {"id": "1447515480079335554", "name": "Donowall", "animated": True}}
            ),
            "<a:Donowall:1447515480079335554>",
        )
        self.assertEqual(
            parse_emoji(
                None, {"emoji": {"id": "1425049288240529468", "name": "joecool", "animated": False}}
            ),
            "<:joecool:1425049288240529468>",
        )

    def test_unusable_emoji_dict_raises(self):
        with self.assertRaisesRegex(ValueError, "emojiId"):
            parse_emoji(None, {"emoji": {"emojiId": "1447515480079335554"}})
        with self.assertRaisesRegex(ValueError, "emoji"):
            parse_emoji(None, {"emojiId": "1447515480079335554"})

    def test_absent_emoji_stays_none(self):
        for value in (None, {}, "", "   "):
            with self.subTest(value=value):
                self.assertIsNone(parse_emoji(None, value))


class P1OnboardingDefaultChannels(unittest.TestCase):
    def test_read_emits_snowflakes_plus_display_names(self):
        onboarding = SimpleNamespace(
            enabled=True,
            mode=None,
            default_channels=[
                SimpleNamespace(id=111, name="gateway"),
                SimpleNamespace(id=222, name="rules"),
            ],
            prompts=[],
        )
        payload = _serialize_onboarding(onboarding)

        self.assertEqual(payload["defaultChannelIds"], ["111", "222"])
        self.assertEqual(payload["defaultChannels"], ["gateway", "rules"])


class P1OnboardingRoundTrip(unittest.IsolatedAsyncioTestCase):
    async def test_read_payload_writes_back_the_same_ids(self):
        channels = [
            FakeChannel(id=111, name="gateway"),
            FakeChannel(id=222, name="rules"),
        ]
        guild = FakeGuild(channels)
        current = {
            "prompts": [],
            "default_channel_ids": ["111", "222"],
            "enabled": True,
            "mode": 0,
        }
        read_payload = _serialize_onboarding(
            SimpleNamespace(
                enabled=True,
                mode=None,
                default_channels=channels,
                prompts=[],
            )
        )

        await handle_update_guild_onboarding(
            {"server_id": "1", "onboarding": read_payload},
            _deps(guild, current),
        )

        _, kwargs = guild.edit_calls[0]
        self.assertEqual([c.id for c in kwargs["default_channels"]], [111, 222])

    async def test_display_names_also_resolve(self):
        guild = FakeGuild([FakeChannel(id=111, name="gateway")])
        await handle_update_guild_onboarding(
            {"server_id": "1", "onboarding": {"defaultChannels": ["gateway"]}},
            _deps(guild),
        )
        _, kwargs = guild.edit_calls[0]
        self.assertEqual([c.id for c in kwargs["default_channels"]], [111])

    async def test_unknown_and_ambiguous_channel_names_are_rejected(self):
        guild = FakeGuild(
            [FakeChannel(id=111, name="dup"), FakeChannel(id=222, name="dup")]
        )
        with self.assertRaisesRegex(ValueError, "matches 2 channels"):
            await handle_update_guild_onboarding(
                {"server_id": "1", "onboarding": {"defaultChannels": ["dup"]}},
                _deps(guild),
            )
        with self.assertRaisesRegex(ValueError, "not found in server"):
            await handle_update_guild_onboarding(
                {"server_id": "1", "onboarding": {"defaultChannels": ["nope"]}},
                _deps(guild),
            )


class P2WelcomeScreenRoundTrip(unittest.IsolatedAsyncioTestCase):
    def _screen(self):
        return SimpleNamespace(
            description="welcome",
            enabled=True,
            welcome_channels=[
                SimpleNamespace(
                    channel=SimpleNamespace(id=111, name="gateway"),
                    description="read me",
                    emoji=CUSTOM_EMOJI,
                )
            ],
        )

    def test_read_keeps_the_custom_emoji_id(self):
        row = _serialize_welcome_screen(self._screen())["welcomeChannels"][0]
        self.assertEqual(row["emoji"], "Herta_Kurukuru")
        self.assertEqual(row["emojiId"], "1447691293911285780")
        self.assertTrue(row["emojiAnimated"])

    async def test_read_payload_writes_back_the_same_emoji(self):
        guild = FakeGuild([FakeChannel(id=111, name="gateway")])
        read_payload = _serialize_welcome_screen(self._screen())

        await handle_update_guild_welcome_screen(
            {"server_id": "1", "welcome_screen": read_payload},
            _deps(guild),
        )

        _, kwargs = guild.edit_calls[0]
        welcome_channel = kwargs["welcome_channels"][0]
        self.assertEqual(welcome_channel.channel.id, 111)
        self.assertEqual(welcome_channel.emoji.id, 1447691293911285780)
        self.assertEqual(welcome_channel.emoji.name, "Herta_Kurukuru")
        self.assertTrue(welcome_channel.emoji.animated)
        # the wire payload discord.py builds must carry the id, not just the name
        self.assertEqual(welcome_channel.to_dict()["emoji_id"], 1447691293911285780)
        self.assertEqual(welcome_channel.to_dict()["emoji_name"], "Herta_Kurukuru")

    async def test_bare_unicode_emoji_still_works(self):
        guild = FakeGuild([FakeChannel(id=111, name="gateway")])
        await handle_update_guild_welcome_screen(
            {
                "server_id": "1",
                "welcome_screen": {
                    "welcome_channels": [
                        {"channel_id": "111", "description": "hi", "emoji": "🔥"}
                    ]
                },
            },
            _deps(guild),
        )
        _, kwargs = guild.edit_calls[0]
        self.assertEqual(str(kwargs["welcome_channels"][0].emoji), "🔥")


class P3ForumTagRoundTrip(unittest.IsolatedAsyncioTestCase):
    def _forum(self):
        return FakeChannel(
            id=30,
            name="knowledge-base",
            type=discord.ChannelType.forum,
            position=1,
            category_id=None,
            topic="topics",
            nsfw=False,
            slowmode_delay=0,
            default_auto_archive_duration=1440,
            default_sort_order=None,
            default_layout=None,
            available_tags=[
                SimpleNamespace(id=7, name="help", emoji=CUSTOM_EMOJI, moderated=True)
            ],
            default_reaction_emoji=CUSTOM_EMOJI,
            overwrites={},
        )

    def test_read_exposes_tags_and_default_reaction_with_ids(self):
        payload = json.loads(
            asyncio_run(
                handle_get_channels_structured(
                    {"server_id": "1"},
                    {"gateway": FakeGateway(FakeGuild([self._forum()]))},
                )
            )[0].text
        )
        row = payload["channels"][0]
        tag = row["availableTags"][0]
        self.assertEqual(tag["name"], "help")
        self.assertEqual(tag["emojiId"], "1447691293911285780")
        self.assertTrue(tag["emojiAnimated"])
        self.assertTrue(tag["moderated"])
        self.assertEqual(row["defaultReactionEmoji"]["emojiId"], "1447691293911285780")

    def test_read_payload_writes_back_the_same_tag_and_reaction(self):
        channel = self._forum()
        read_row = {
            "availableTags": [_serialize_forum_tag(channel.available_tags[0])],
            "defaultReactionEmoji": {
                "emoji": "Herta_Kurukuru",
                "emojiId": "1447691293911285780",
                "emojiAnimated": True,
            },
        }
        asyncio_run(
            handle_update_forum_channel(
                {"server_id": "1", "channel_id": "30", **read_row},
                {"gateway": FakeGateway(FakeGuild([channel]))},
            )
        )
        kwargs = channel.edit_calls[0]
        tag = kwargs["available_tags"][0]
        self.assertIsInstance(tag, discord.ForumTag)
        self.assertEqual(tag.name, "help")
        self.assertTrue(tag.moderated)
        self.assertEqual(tag.emoji.id, 1447691293911285780)
        self.assertTrue(tag.emoji.animated)
        self.assertEqual(
            kwargs["default_reaction_emoji"], "<a:Herta_Kurukuru:1447691293911285780>"
        )

    def test_name_from_the_read_payload_resolves_against_guild_emojis(self):
        channel = self._forum()
        guild = FakeGuild([channel])
        guild.emojis = [CUSTOM_EMOJI]
        asyncio_run(
            handle_update_forum_channel(
                {
                    "server_id": "1",
                    "channel_id": "30",
                    "availableTags": [
                        {"name": "help", "emoji": "Herta_Kurukuru", "moderated": False}
                    ],
                    "defaultReactionEmoji": "Herta_Kurukuru",
                },
                {"gateway": FakeGateway(guild)},
            )
        )
        kwargs = channel.edit_calls[0]
        self.assertEqual(kwargs["available_tags"][0].emoji.id, 1447691293911285780)
        self.assertEqual(
            kwargs["default_reaction_emoji"], "<a:Herta_Kurukuru:1447691293911285780>"
        )

    def test_tag_validation(self):
        channel = self._forum()
        with self.assertRaisesRegex(ValueError, "name is required"):
            asyncio_run(
                handle_update_forum_channel(
                    {"server_id": "1", "channel_id": "30", "availableTags": [{}]},
                    {"gateway": FakeGateway(FakeGuild([channel]))},
                )
            )
        with self.assertRaisesRegex(ValueError, "Discord allows 20"):
            asyncio_run(
                handle_update_forum_channel(
                    {
                        "server_id": "1",
                        "channel_id": "30",
                        "availableTags": [{"name": "x" * 21}],
                    },
                    {"gateway": FakeGateway(FakeGuild([channel]))},
                )
            )


def asyncio_run(coro):
    """Run a coroutine from a sync test (no event loop is active there)."""
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
