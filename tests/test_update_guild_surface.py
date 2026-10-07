import base64
import json
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

import discord  # noqa: E402
from discord_mcp.core.images import load_image_bytes  # noqa: E402
from discord_mcp.tools.handlers.server_info import (  # noqa: E402
    handle_update_guild,
)


class FakeChannel:
    def __init__(self, channel_id, name):
        self.id = channel_id
        self.name = name


class FakeGuild:
    id = 1
    name = "Guild"
    description = None
    preferred_locale = "en-US"
    verification_level = discord.VerificationLevel.none
    explicit_content_filter = discord.ContentFilter.disabled
    default_notifications = discord.NotificationLevel.only_mentions
    features = ["COMMUNITY"]

    def __init__(self):
        self.channels = [
            FakeChannel(10, "general"),
            FakeChannel(11, "general"),  # duplicate on purpose: ambiguity check
            FakeChannel(12, "general-2"),
        ]
        self.edits = []

    def get_channel(self, channel_id):
        return next((c for c in self.channels if c.id == channel_id), None)

    async def edit(self, **kwargs):
        self.edits.append(kwargs)
        if "description" in kwargs:
            self.description = kwargs["description"]


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, server_id=None):
        return self.guild


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AARAA"
    "B/2kCxgAAAABJRU5ErkJggg=="
)


class ImageLoaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_data_uri_is_decoded(self):
        data = await load_image_bytes(
            "data:image/png;base64," + base64.b64encode(PNG).decode(), "icon"
        )
        self.assertEqual(data, PNG)

    async def test_null_clears_and_bad_value_is_rejected(self):
        self.assertIsNone(await load_image_bytes(None, "icon"))
        with self.assertRaisesRegex(ValueError, "not an http"):
            await load_image_bytes("nope-not-a-file", "banner")
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            await load_image_bytes("data:image/png;base64,!!!!", "icon")

    async def test_oversized_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Discord's limit"):
            await load_image_bytes(b"x" * (10 * 1024 * 1024 + 1), "icon")


class UpdateGuildSurfaceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.guild = FakeGuild()
        self.deps = {"gateway": FakeGateway(self.guild)}

    async def test_bool_enum_and_string_fields_are_mapped(self):
        payload = json.loads(
            (
                await handle_update_guild(
                    {
                        "server_id": "1",
                        "name": "New name",
                        "description": "hello",
                        "preferred_locale": "vi",
                        "verification_level": "high",
                        "explicit_content_filter": "all_members",
                        "default_notifications": "all_messages",
                        "mfa_level": "elevated",
                        "community": True,
                        "discoverable": False,
                        "premium_progress_bar_enabled": True,
                        "raid_alerts_disabled": True,
                        "invites_disabled": False,
                        "widget_enabled": False,
                        "reason": "tune",
                    },
                    self.deps,
                )
            )[0].text
        )
        kwargs = self.guild.edits[0]
        self.assertEqual(kwargs["name"], "New name")
        self.assertEqual(kwargs["description"], "hello")
        self.assertEqual(kwargs["preferred_locale"], "vi")
        self.assertEqual(kwargs["verification_level"], discord.VerificationLevel.high)
        self.assertEqual(
            kwargs["explicit_content_filter"], discord.ContentFilter.all_members
        )
        self.assertEqual(
            kwargs["default_notifications"], discord.NotificationLevel.all_messages
        )
        self.assertEqual(kwargs["mfa_level"], discord.MFALevel.require_2fa)
        self.assertTrue(kwargs["community"])
        self.assertFalse(kwargs["discoverable"])
        self.assertTrue(kwargs["premium_progress_bar_enabled"])
        self.assertTrue(kwargs["raid_alerts_disabled"])
        self.assertEqual(kwargs["reason"], "tune")
        self.assertEqual(payload["status"], "applied")
        self.assertIn("preferred_locale", payload["applied_fields"])

    async def test_channel_fields_accept_ids_and_names(self):
        await handle_update_guild(
            {
                "server_id": "1",
                "afk_channel": "10",
                "system_channel": "general-2",
                "rules_channel": None,
                "afk_timeout": 300,
            },
            self.deps,
        )
        kwargs = self.guild.edits[0]
        self.assertEqual(kwargs["afk_channel"].id, 10)
        self.assertEqual(kwargs["system_channel"].name, "general-2")
        self.assertIsNone(kwargs["rules_channel"])
        self.assertEqual(kwargs["afk_timeout"], 300)

    async def test_ambiguous_channel_name_and_bad_timeout_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            await handle_update_guild(
                {"server_id": "1", "system_channel": "general"}, self.deps
            )
        with self.assertRaisesRegex(ValueError, "afk_timeout must be one of"):
            await handle_update_guild({"server_id": "1", "afk_timeout": 42}, self.deps)

    async def test_system_channel_flags_accept_names_and_int(self):
        await handle_update_guild(
            {"server_id": "1", "system_channel_flags": ["emoji_added"]},
            self.deps,
        )
        flags = self.guild.edits[0]["system_channel_flags"]
        self.assertTrue(flags.emoji_added)

        await handle_update_guild(
            {"server_id": "1", "system_channel_flags": 0}, self.deps
        )
        self.assertEqual(self.guild.edits[1]["system_channel_flags"].value, 0)

    async def test_image_field_and_datetime_fields(self):
        await handle_update_guild(
            {
                "server_id": "1",
                "icon": "data:image/png;base64," + base64.b64encode(PNG).decode(),
                "dms_disabled_until": "2026-12-01T00:00:00Z",
                "invites_disabled_until": None,
            },
            self.deps,
        )
        kwargs = self.guild.edits[0]
        self.assertEqual(kwargs["icon"], PNG)
        self.assertEqual(kwargs["dms_disabled_until"].year, 2026)
        self.assertIsNone(kwargs["invites_disabled_until"])

    async def test_unknown_fields_are_rejected_with_the_supported_list(self):
        with self.assertRaisesRegex(ValueError, "unsupported_fields: traits"):
            await handle_update_guild(
                {"server_id": "1", "traits": ["Chữa lành"]}, self.deps
            )

    async def test_empty_call_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nothing to update"):
            await handle_update_guild({"server_id": "1"}, self.deps)

    async def test_forbidden_explains_manage_guild_and_feature_gates(self):
        class ForbiddenGuild(FakeGuild):
            async def edit(self, **kwargs):
                response = type("R", (), {"status": 403, "reason": "Forbidden"})()
                raise discord.Forbidden(
                    response, {"code": 50013, "message": "Missing Permissions"}
                )

        deps = {"gateway": FakeGateway(ForbiddenGuild())}
        with self.assertRaisesRegex(ValueError, "MANAGE_GUILD") as ctx:
            await handle_update_guild({"server_id": "1", "icon": None}, deps)
        self.assertIn("matching guild feature", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
