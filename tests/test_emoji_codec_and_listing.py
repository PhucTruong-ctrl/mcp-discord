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

from discord_mcp.core.emoji import (  # noqa: E402
    emoji_payload,
    emoji_token,
    parse_emoji,
    serialize_emoji,
)
from discord_mcp.tools.handlers.emoji import (  # noqa: E402
    handle_list_guild_emojis,
)


def make_emoji(emoji_id="1", name="wave", animated=False, roles=None, **extra):
    """Custom emoji stand-in exposing only the attributes the codec reads."""
    attrs = dict(
        id=emoji_id,
        name=name,
        animated=animated,
        available=True,
        managed=False,
        require_colons=True,
        url=f"https://cdn.example/{emoji_id}.png",
        roles=roles,
    )
    attrs.update(extra)
    return SimpleNamespace(**attrs)


def make_sticker(sticker_id="99", name="pepe", **extra):
    attrs = dict(
        id=sticker_id,
        name=name,
        description="a sticker",
        type=2,
        format=1,
        available=True,
        tags="frog",
    )
    attrs.update(extra)
    return SimpleNamespace(**attrs)


class FakeGuild:
    def __init__(self, guild_id="555", emojis=(), stickers=()):
        self.id = guild_id
        self.emojis = list(emojis)
        self.stickers = list(stickers)


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, server_id):
        return self.guild


def payload_of(result):
    return json.loads(result[0].text)


class SerializeEmojiTests(unittest.TestCase):
    def test_unicode_string(self):
        self.assertEqual(serialize_emoji("\U0001f525"), ("\U0001f525", None, False))

    def test_custom_animated_emoji(self):
        self.assertEqual(
            serialize_emoji(make_emoji("7", "wave", True)), ("wave", "7", True)
        )

    def test_none(self):
        self.assertEqual(serialize_emoji(None), (None, None, False))


class EmojiPayloadTests(unittest.TestCase):
    def test_unicode_shape(self):
        self.assertEqual(
            emoji_payload("\U0001f525"),
            {"emoji": "\U0001f525", "emojiId": None, "emojiAnimated": False},
        )

    def test_custom_animated_shape(self):
        self.assertEqual(
            emoji_payload(make_emoji("7", "wave", True)),
            {"emoji": "wave", "emojiId": "7", "emojiAnimated": True},
        )


class EmojiTokenTests(unittest.TestCase):
    def test_unicode_passes_through(self):
        self.assertEqual(emoji_token("\U0001f525", None), "\U0001f525")

    def test_id_without_name_returns_name_only(self):
        self.assertEqual(emoji_token("wave", None), "wave")

    def test_animated_token(self):
        self.assertEqual(emoji_token("wave", "7", True), "<a:wave:7>")

    def test_static_token(self):
        self.assertEqual(emoji_token("wave", "7", False), "<:wave:7>")


class ParseEmojiTests(unittest.TestCase):
    def setUp(self):
        self.guild = FakeGuild(emojis=[make_emoji("7", "wave", True)], stickers=[])

    def test_none(self):
        self.assertIsNone(parse_emoji(self.guild, None))

    def test_unicode(self):
        self.assertEqual(parse_emoji(self.guild, "\U0001f525"), "\U0001f525")

    def test_existing_guild_emoji_name_resolves_to_token(self):
        self.assertEqual(parse_emoji(self.guild, "wave"), "<a:wave:7>")

    def test_explicit_token_passes_through(self):
        self.assertEqual(parse_emoji(self.guild, "<:wave:7>"), "<:wave:7>")

    def test_read_shape_dict(self):
        self.assertEqual(
            parse_emoji(
                self.guild, {"emoji": "wave", "emojiId": "7", "emojiAnimated": True}
            ),
            "<a:wave:7>",
        )

    def test_write_shape_dict(self):
        self.assertEqual(
            parse_emoji(self.guild, {"name": "wave", "id": "7"}), "<:wave:7>"
        )

    def test_unknown_bare_name_stays(self):
        self.assertEqual(parse_emoji(self.guild, "nope"), "nope")

    def test_empty_string_is_none(self):
        self.assertIsNone(parse_emoji(self.guild, ""))


class ListGuildEmojisTests(unittest.IsolatedAsyncioTestCase):
    def _deps(self):
        role = SimpleNamespace(id="42")
        self.gif = make_emoji("1", "party", animated=True, roles=[role])
        self.static = make_emoji("2", "wave", animated=False, roles=[])
        self.sticker = make_sticker("99", "pepe")
        guild = FakeGuild(
            guild_id="555", emojis=[self.static, self.gif], stickers=[self.sticker]
        )
        return {"gateway": FakeGateway(guild)}

    async def test_default_lists_all_emojis_and_no_stickers(self):
        data = payload_of(
            await handle_list_guild_emojis({"server_id": "555"}, self._deps())
        )
        self.assertEqual(data["emojiCount"], 2)
        self.assertEqual([row["name"] for row in data["emojis"]], ["party", "wave"])
        self.assertEqual(data["emojiCount"], len(data["emojis"]))
        self.assertEqual(data["stickerCount"], 0)
        self.assertEqual(data["stickers"], [])

    async def test_include_stickers_lists_sticker(self):
        deps = self._deps()
        data = payload_of(
            await handle_list_guild_emojis(
                {"server_id": "555", "include_stickers": True}, deps
            )
        )
        self.assertEqual(data["stickerCount"], 1)
        self.assertEqual(data["stickers"][0]["id"], "99")
        self.assertEqual(data["stickers"][0]["name"], "pepe")

    async def test_name_contains_filters_case_insensitively(self):
        deps = self._deps()
        data = payload_of(
            await handle_list_guild_emojis(
                {"server_id": "555", "name_contains": "PAR"}, deps
            )
        )
        self.assertEqual(data["emojiCount"], 1)
        self.assertEqual(data["emojis"][0]["name"], "party")

    async def test_animated_token_matches_reaction_tool_form(self):
        data = payload_of(
            await handle_list_guild_emojis({"server_id": "555"}, self._deps())
        )
        animated = next(row for row in data["emojis"] if row["name"] == "party")
        self.assertTrue(animated["animated"])
        self.assertEqual(animated["token"], "<a:party:1>")
        self.assertEqual(animated["id"], "1")
        self.assertEqual(animated["roleIds"], ["42"])


if __name__ == "__main__":
    unittest.main()
