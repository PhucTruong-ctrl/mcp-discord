import json
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")

from discord_mcp.core.permissions import effective_permissions  # noqa: E402
from discord_mcp.tools.handlers.permission_intel import (  # noqa: E402
    handle_compute_member_permissions,
    handle_get_role_permissions,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER  # noqa: E402
from discord_mcp.tools.schemas import compose_tool_registry  # noqa: E402


MENTION_EVERYONE = 1 << 17
SEND_MESSAGES = 1 << 11


class FakeFlag(int):
    """Stands in for discord.Permissions: the helpers read ``.value`` or the int."""

    @property
    def value(self):
        return int(self)


class FakeOverwrite:
    def __init__(self, allow=0, deny=0):
        self._allow = FakeFlag(allow)
        self._deny = FakeFlag(deny)

    def pair(self):
        return self._allow, self._deny


class FakeTarget:
    def __init__(self, target_id, name):
        self.id = target_id
        self.name = name


class FakeRole:
    def __init__(self, role_id, name, position, permissions=0, hoist=False):
        self.id = role_id
        self.name = name
        self.position = position
        self.permissions = FakeFlag(permissions)
        self.hoist = hoist
        self.mentionable = False
        self.managed = False


class FakeMember:
    def __init__(self, member_id, name, roles):
        self.id = member_id
        self.display_name = name
        self.roles = roles


class FakeChannel:
    def __init__(self, channel_id, name, overwrites=None, category=None):
        self.id = channel_id
        self.name = name
        self.type = "text"
        self.category_id = getattr(category, "id", None)
        self.category = category
        self.overwrites = overwrites or {}
        self.parent = None


class FakeGuild:
    def __init__(self):
        self.id = 1
        self.name = "Guild"
        self.owner_id = 99
        self.default_role = FakeRole(1, "@everyone", 0, permissions=SEND_MESSAGES)
        self.giver = FakeRole(2, "🍀 Thành viên", 5, permissions=MENTION_EVERYONE)
        self.admin = FakeRole(3, "Admin", 10, permissions=8)
        self.roles = [self.default_role, self.giver, self.admin]
        self.members = {
            10: FakeMember(10, "shiro", [self.default_role, self.giver]),
            11: FakeMember(11, "root", [self.default_role, self.admin]),
            99: FakeMember(99, "owner", [self.default_role]),
        }
        self.channel = FakeChannel(
            500,
            "general",
            overwrites={
                FakeTarget(1, "@everyone"): FakeOverwrite(
                    deny=MENTION_EVERYONE | SEND_MESSAGES
                ),
                FakeTarget(2, "🍀 Thành viên"): FakeOverwrite(allow=MENTION_EVERYONE),
            },
        )

    def get_role(self, role_id):
        return next((role for role in self.roles if role.id == role_id), None)

    def get_channel_or_thread(self, channel_id):
        return self.channel if int(channel_id) == self.channel.id else None

    async def fetch_member(self, user_id):
        return self.members[user_id]


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def resolve_guild(self, _server_id):
        return self.guild

    async def fetch_channel(self, _channel_id):
        return None


class PermissionIntelRegistryTests(unittest.TestCase):
    def test_tools_registered_in_schema_and_router(self):
        names = {tool.name for tool in compose_tool_registry()}
        for tool_name in ("get_role_permissions", "compute_member_permissions"):
            self.assertIn(tool_name, names)
            self.assertIn(tool_name, TOOL_ROUTER)


class PermissionIntelHandlerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.guild = FakeGuild()
        self.deps = {"gateway": FakeGateway(self.guild)}

    async def test_list_roles_exposes_permission_bitfields(self):
        result = await handle_get_role_permissions({"server_id": "1"}, self.deps)
        payload = json.loads(result[0].text)

        self.assertEqual(payload["roleCount"], 3)
        roles = {role["name"]: role for role in payload["roles"]}
        self.assertEqual(roles["🍀 Thành viên"]["permissions"], MENTION_EVERYONE)
        self.assertEqual(
            roles["🍀 Thành viên"]["permissionNames"], ["mention_everyone"]
        )
        self.assertEqual(roles["Admin"]["permissionNames"], ["administrator"])

    async def test_single_role_lookup(self):
        result = await handle_get_role_permissions(
            {"server_id": "1", "role_id": "2"}, self.deps
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["roles"][0]["id"], "2")

        with self.assertRaisesRegex(ValueError, "not found"):
            await handle_get_role_permissions(
                {"server_id": "1", "role_id": "404"}, self.deps
            )

    async def test_member_permissions_source_is_the_role_that_grants_the_bit(self):
        result = await handle_compute_member_permissions(
            {"server_id": "1", "member_id": "10"}, self.deps
        )
        payload = json.loads(result[0].text)

        self.assertTrue(payload["effective"]["permissions"] & MENTION_EVERYONE)
        self.assertEqual(payload["sources"]["mention_everyone"], "base_role")
        self.assertEqual(payload["channelId"], None)

    async def test_channel_overwrites_decide_the_bit(self):
        result = await handle_compute_member_permissions(
            {"server_id": "1", "member_id": "10", "channel_id": "500"}, self.deps
        )
        payload = json.loads(result[0].text)

        # @everyone deny wins first, then the member's role allow is applied
        self.assertTrue(payload["effective"]["permissions"] & MENTION_EVERYONE)
        self.assertEqual(
            payload["sources"]["mention_everyone"], "role_overwrite:2:allow"
        )
        self.assertEqual(payload["channelId"], "500")

    async def test_administrator_short_circuits(self):
        result = await handle_compute_member_permissions(
            {"server_id": "1", "member_id": "11"}, self.deps
        )
        payload = json.loads(result[0].text)

        self.assertTrue(payload["isAdministrator"])
        self.assertEqual(payload["sources"]["mention_everyone"], "administrator")

    def test_effective_permissions_matches_discord_py_for_role_only_channels(self):
        guild = self.guild
        resolved = effective_permissions(guild, guild.members[10], guild.channel)
        # base: SEND_MESSAGES from @everyone + MENTION_EVERYONE from the role
        self.assertEqual(resolved["base"], SEND_MESSAGES | MENTION_EVERYONE)
        # the channel denies both, then the member's role re-allows MENTION_EVERYONE
        self.assertFalse(resolved["effective"] & SEND_MESSAGES)
        self.assertTrue(resolved["effective"] & MENTION_EVERYONE)
        self.assertEqual(
            resolved["sources"]["send_messages"], "everyone_overwrite:deny"
        )
        self.assertEqual(
            resolved["sources"]["mention_everyone"], "role_overwrite:2:allow"
        )


if __name__ == "__main__":
    unittest.main()
