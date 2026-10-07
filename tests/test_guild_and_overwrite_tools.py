"""Tests for the guild-settings and channel-overwrite write tools."""

import datetime
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock

import discord


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

from discord_mcp.tools.handlers.inventory import (
    handle_remove_channel_permission_overwrite,
    handle_set_channel_permission_overwrite,
    _permission_mask,
)
from discord_mcp.tools.handlers.server_info import (
    handle_get_server_info,
    handle_update_guild,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER


def _payload(result):
    return json.loads(result[0].text)


def _fake_guild():
    guild = AsyncMock()
    guild.id = 999
    guild.name = "Test Guild"
    default_role = AsyncMock()
    default_role.id = 999
    default_role.name = "@everyone"
    guild.default_role = default_role
    guild.get_role = lambda role_id: (
        _FakeRole(role_id, "Moderator") if role_id == 111 else None
    )
    guild.fetch_roles = AsyncMock(return_value=[_FakeRole(111, "Moderator")])
    guild.get_member = lambda member_id: None
    guild.fetch_member = AsyncMock(side_effect=discord.NotFound)
    guild.description = None
    guild.verification_level = discord.VerificationLevel.none
    guild.explicit_content_filter = discord.ContentFilter.disabled
    return guild


class _FakeRole:
    def __init__(self, role_id, name):
        self.id = role_id
        self.name = name


class _FakeChannel:
    """Channel stand-in recording set_permissions calls."""

    def __init__(self, guild):
        self.id = 555
        self.name = "general"
        self.guild = guild
        self.calls = []
        self._overwrites = {}

    async def set_permissions(self, target, *, overwrite=None, reason=None):
        self.calls.append((target.id, overwrite, reason))
        if overwrite is None:
            self._overwrites.pop(target.id, None)
        else:
            self._overwrites[target.id] = overwrite


class PermissionMaskTests(unittest.TestCase):
    def test_names_and_bits_combine(self):
        mask = _permission_mask(["send_messages", "add-reactions", 8], "deny")
        self.assertEqual(mask, 0x800 | 0x40 | 8)

    def test_unknown_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown deny permission"):
            _permission_mask(["not_a_permission"], "deny")

    def test_non_sequence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be an array"):
            _permission_mask("send_messages", "allow")


class UpdateGuildTests(unittest.IsolatedAsyncioTestCase):
    def _deps(self):
        guild = _fake_guild()
        guild.edit = AsyncMock()
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        return guild, {"gateway": gateway}

    async def test_updates_description_and_verification_level(self):
        guild, deps = self._deps()
        guild.verification_level = discord.VerificationLevel.medium
        guild.explicit_content_filter = discord.ContentFilter.all_members
        guild.description = "hi"

        result = await handle_update_guild(
            {
                "server_id": "999",
                "description": "hi",
                "verification_level": "medium",
                "reason": "audit",
            },
            deps,
        )

        guild.edit.assert_awaited_once_with(
            reason="audit",
            description="hi",
            verification_level=discord.VerificationLevel.medium,
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["verification_level"], "medium")

    async def test_numeric_levels_and_aliases(self):
        guild, deps = self._deps()
        await handle_update_guild({"server_id": "999", "verification_level": 4}, deps)
        self.assertEqual(
            guild.edit.call_args.kwargs["verification_level"],
            discord.VerificationLevel.highest,
        )

        guild.edit.reset_mock()
        await handle_update_guild(
            {"server_id": "999", "explicit_content_filter": "all_members"}, deps
        )
        self.assertEqual(
            guild.edit.call_args.kwargs["explicit_content_filter"],
            discord.ContentFilter.all_members,
        )

    async def test_requires_something_to_update(self):
        _, deps = self._deps()
        with self.assertRaisesRegex(ValueError, "nothing to update"):
            await handle_update_guild({"server_id": "999"}, deps)

    async def test_rejects_unknown_level(self):
        _, deps = self._deps()
        with self.assertRaisesRegex(ValueError, "unknown verification_level"):
            await handle_update_guild(
                {"server_id": "999", "verification_level": "paranoid"}, deps
            )

    async def test_description_null_clears(self):
        guild, deps = self._deps()
        await handle_update_guild({"server_id": "999", "description": None}, deps)
        self.assertIsNone(guild.edit.call_args.kwargs["description"])


class ChannelOverwriteTests(unittest.IsolatedAsyncioTestCase):
    def _deps(self):
        guild = _fake_guild()
        channel = _FakeChannel(guild)
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        gateway.fetch_channel = AsyncMock(return_value=channel)
        return channel, {"gateway": gateway}

    async def test_set_overwrite_for_role(self):
        channel, deps = self._deps()
        result = await handle_set_channel_permission_overwrite(
            {
                "channel_id": "555",
                "target_id": "111",
                "deny": ["send_messages"],
                "allow": ["read_messages"],
                "reason": "lock",
            },
            deps,
        )
        target_id, overwrite, reason = channel.calls[0]
        self.assertEqual(target_id, 111)
        self.assertEqual(reason, "lock")
        allow, deny = overwrite.pair()
        self.assertEqual(allow.value, 0x400)
        self.assertEqual(deny.value, 0x800)
        self.assertEqual(_payload(result)["status"], "applied")

    async def test_set_overwrite_for_everyone_role(self):
        channel, deps = self._deps()
        await handle_set_channel_permission_overwrite(
            {"channel_id": "555", "target_id": "999", "deny": ["read_messages"]},
            deps,
        )
        self.assertEqual(channel.calls[0][0], 999)

    async def test_empty_allow_and_deny_is_rejected(self):
        channel, deps = self._deps()
        with self.assertRaisesRegex(ValueError, "both empty"):
            await handle_set_channel_permission_overwrite(
                {"channel_id": "555", "target_id": "111"}, deps
            )
        self.assertEqual(channel.calls, [])

    async def test_unknown_member_requires_target_type(self):
        _, deps = self._deps()
        with self.assertRaisesRegex(ValueError, "not a cached role or member"):
            await handle_set_channel_permission_overwrite(
                {"channel_id": "555", "target_id": "1234567890"}, deps
            )

    async def test_invalid_target_type_is_rejected(self):
        _, deps = self._deps()
        with self.assertRaisesRegex(ValueError, "target_type must be"):
            await handle_set_channel_permission_overwrite(
                {
                    "channel_id": "555",
                    "target_id": "111",
                    "target_type": "guild",
                    "deny": ["send_messages"],
                },
                deps,
            )

    async def test_remove_overwrite(self):
        channel, deps = self._deps()
        result = await handle_remove_channel_permission_overwrite(
            {"channel_id": "555", "target_id": "111", "reason": "cleanup"}, deps
        )
        self.assertEqual(channel.calls[0], (111, None, "cleanup"))
        self.assertEqual(_payload(result)["status"], "applied")

    def test_tools_are_registered(self):
        for name in (
            "update_guild",
            "set_channel_permission_overwrite",
            "remove_channel_permission_overwrite",
        ):
            self.assertIn(name, TOOL_ROUTER)


class ServerInfoFreshnessTests(unittest.IsolatedAsyncioTestCase):
    """get_server_info must read a freshly fetched guild, not the gateway cache."""

    class _FreshGuild:
        name = "Bên Hiên Nhà"
        id = 999
        owner_id = 111
        member_count = 293
        created_at = datetime.datetime(2025, 10, 4, tzinfo=datetime.timezone.utc)
        description = "fresh description"
        verification_level = discord.VerificationLevel.high
        approximate_member_count = None
        premium_tier = 0
        explicit_content_filter = discord.ContentFilter.all_members

    async def test_uses_fetch_guild_and_reports_verification_level(self):
        gateway = AsyncMock()
        gateway.fetch_guild = AsyncMock(return_value=self._FreshGuild())
        gateway.resolve_guild = AsyncMock(
            side_effect=AssertionError("must not read the cached guild")
        )

        text = (
            await handle_get_server_info({"server_id": "999"}, {"gateway": gateway})
        )[0].text

        gateway.fetch_guild.assert_awaited_once_with("999")
        self.assertIn("description: fresh description", text)
        self.assertIn("verification_level: high", text)
        self.assertIn("member_count: 293", text)

    async def test_member_count_falls_back_to_approximate(self):
        fresh = self._FreshGuild()
        fresh.member_count = None
        fresh.approximate_member_count = 291
        gateway = AsyncMock()
        gateway.fetch_guild = AsyncMock(return_value=fresh)

        text = (
            await handle_get_server_info({"server_id": "999"}, {"gateway": gateway})
        )[0].text

        self.assertIn("member_count: 291", text)


if __name__ == "__main__":
    unittest.main()
