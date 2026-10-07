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
from discord_mcp.tools.handlers.member_admin import (  # noqa: E402
    handle_set_member_nickname,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER  # noqa: E402
from discord_mcp.tools.schemas import compose_tool_registry  # noqa: E402


class FakeMember:
    id = 55
    nick = "old nick"

    def __init__(self, forbidden=False):
        self.calls = []
        self.forbidden = forbidden

    def __str__(self):
        return "nickname-user"

    @property
    def display_name(self):
        return self.nick or "global-name"

    async def edit(self, *, nick, reason=None):
        if self.forbidden:
            response = type("R", (), {"status": 403, "reason": "Forbidden"})()
            data = {"code": 50013, "message": "Missing Permissions"}
            raise discord.Forbidden(response, data)
        self.calls.append((nick, reason))
        self.nick = nick


class FakeGuild:
    id = 1
    name = "Guild"

    def __init__(self, member):
        self.member = member
        self.me = type("Me", (), {"id": 999})()

    async def fetch_member(self, member_id):
        if member_id != self.member.id:
            raise ValueError("Unknown Member")
        return self.member


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, server_id=None):
        return self.guild


class MemberAdminRegistryTests(unittest.TestCase):
    def test_tool_registered_in_schema_and_router(self):
        names = {tool.name for tool in compose_tool_registry()}
        self.assertIn("set_member_nickname", names)
        self.assertIn("set_member_nickname", TOOL_ROUTER)


class SetMemberNicknameTests(unittest.IsolatedAsyncioTestCase):
    def _deps(self, forbidden=False):
        member = FakeMember(forbidden=forbidden)
        return {"gateway": FakeGateway(FakeGuild(member))}, member

    async def test_sets_nickname_with_reason(self):
        deps, member = self._deps()
        payload = json.loads(
            (
                await handle_set_member_nickname(
                    {
                        "server_id": "1",
                        "member_id": "55",
                        "nickname": "new nick",
                        "reason": "requested",
                    },
                    deps,
                )
            )[0].text
        )

        self.assertEqual(payload["status"], "executed")
        self.assertEqual(payload["previousNickname"], "old nick")
        self.assertEqual(payload["nickname"], "new nick")
        self.assertFalse(payload["nicknameRemoved"])
        self.assertEqual(payload["displayName"], "new nick")
        self.assertEqual(member.calls, [("new nick", "requested")])

    async def test_empty_nickname_removes_it(self):
        deps, member = self._deps()
        payload = json.loads(
            (
                await handle_set_member_nickname(
                    {"server_id": "1", "member_id": "55", "nickname": ""}, deps
                )
            )[0].text
        )

        self.assertTrue(payload["nicknameRemoved"])
        self.assertIsNone(payload["nickname"])
        # discord.py removes the nickname when nick is None
        self.assertEqual(member.calls, [(None, None)])

    async def test_nickname_longer_than_32_chars_is_rejected(self):
        deps, member = self._deps()
        with self.assertRaisesRegex(ValueError, "at most 32 characters"):
            await handle_set_member_nickname(
                {"server_id": "1", "member_id": "55", "nickname": "x" * 33}, deps
            )
        self.assertEqual(member.calls, [])

    async def test_missing_nickname_argument_is_rejected(self):
        deps, member = self._deps()
        with self.assertRaisesRegex(ValueError, "nickname is required"):
            await handle_set_member_nickname(
                {"server_id": "1", "member_id": "55"}, deps
            )
        self.assertEqual(member.calls, [])

    async def test_unknown_member_is_reported(self):
        deps, _ = self._deps()
        with self.assertRaisesRegex(ValueError, "not found in server"):
            await handle_set_member_nickname(
                {"server_id": "1", "member_id": "404", "nickname": "x"}, deps
            )

    async def test_forbidden_names_the_required_permission(self):
        deps, _ = self._deps(forbidden=True)
        with self.assertRaisesRegex(ValueError, "MANAGE_NICKNAMES"):
            await handle_set_member_nickname(
                {"server_id": "1", "member_id": "55", "nickname": "x"}, deps
            )


if __name__ == "__main__":
    unittest.main()
