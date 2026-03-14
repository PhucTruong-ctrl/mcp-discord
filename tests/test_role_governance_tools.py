import asyncio
import json
import os
import sys
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

from discord_mcp.tools.handlers.role_governance import (  # noqa: E402
    handle_add_roles_bulk,
    handle_create_role,
    handle_delete_role,
    handle_mute_member_role_based,
    handle_permission_drift_check,
    handle_remove_roles_bulk,
    handle_unmute_member_role_based,
    handle_update_role,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER  # noqa: E402
from discord_mcp.tools.schemas import compose_tool_registry  # noqa: E402


@dataclass
class FakeRole:
    id: int
    name: str
    permissions: int = 0
    color: int = 0
    mentionable: bool = False
    hoist: bool = False

    async def edit(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    async def delete(self, reason=None):
        return reason


@dataclass
class FakeMember:
    id: int
    name: str = "member"
    roles: list = field(default_factory=list)
    add_calls: list = field(default_factory=list)
    remove_calls: list = field(default_factory=list)

    async def add_roles(self, *roles, reason=None):
        self.add_calls.append((roles, reason))
        existing = {role.id for role in self.roles}
        for role in roles:
            if role.id not in existing:
                self.roles.append(role)

    async def remove_roles(self, *roles, reason=None):
        self.remove_calls.append((roles, reason))
        remove_ids = {role.id for role in roles}
        self.roles = [role for role in self.roles if role.id not in remove_ids]


class FakeGuild:
    def __init__(self):
        self.id = 1
        self.name = "Guild"
        self.roles = [
            FakeRole(1, "Admin", permissions=8),
            FakeRole(2, "Muted", permissions=0),
            FakeRole(3, "Member", permissions=1),
        ]
        self.members = {
            10: FakeMember(10, roles=[self.roles[2]]),
            11: FakeMember(11, roles=[self.roles[2], self.roles[0]]),
        }
        self.created_roles = []

    def get_role(self, role_id):
        for role in self.roles:
            if role.id == role_id:
                return role
        return None

    async def fetch_member(self, user_id):
        return self.members[user_id]

    async def create_role(self, **kwargs):
        role = FakeRole(id=100 + len(self.created_roles), name=kwargs["name"])
        self.created_roles.append((role, kwargs))
        self.roles.append(role)
        return role


class FakeGateway:
    def __init__(self, guild):
        self.guild = guild

    async def fetch_guild(self, _server_id):
        return self.guild


class RoleGovernanceToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.guild = FakeGuild()
        self.deps = {"gateway": FakeGateway(self.guild)}

    def test_wave5_tools_registered_in_schema_and_router(self):
        names = [tool.name for tool in compose_tool_registry()]
        expected = {
            "create_role",
            "delete_role",
            "update_role",
            "add_roles_bulk",
            "remove_roles_bulk",
            "mute_member_role_based",
            "unmute_member_role_based",
            "permission_drift_check",
        }
        self.assertTrue(expected.issubset(set(names)))
        self.assertTrue(expected.issubset(set(TOOL_ROUTER.keys())))

    async def test_create_update_delete_role_flow(self):
        created = await handle_create_role(
            {
                "server_id": "1",
                "name": "Ops",
                "permissions": 16,
                "color": 123,
                "hoist": True,
                "mentionable": True,
            },
            self.deps,
        )
        payload = json.loads(created[0].text)
        self.assertEqual(payload["roleName"], "Ops")

        role_id = str(payload["roleId"])
        updated = await handle_update_role(
            {
                "server_id": "1",
                "role_id": role_id,
                "name": "Ops2",
                "mentionable": False,
            },
            self.deps,
        )
        self.assertIn("updated", updated[0].text)

        deleted = await handle_delete_role(
            {"server_id": "1", "role_id": role_id, "reason": "cleanup"}, self.deps
        )
        self.assertIn("deleted", deleted[0].text)

    async def test_bulk_add_and_remove_roles(self):
        added = await handle_add_roles_bulk(
            {"server_id": "1", "user_ids": ["10"], "role_ids": ["1"]},
            self.deps,
        )
        add_payload = json.loads(added[0].text)
        self.assertEqual(add_payload["appliedCount"], 1)

        removed = await handle_remove_roles_bulk(
            {"server_id": "1", "user_ids": ["11"], "role_ids": ["1"]},
            self.deps,
        )
        remove_payload = json.loads(removed[0].text)
        self.assertEqual(remove_payload["appliedCount"], 1)

    async def test_mute_and_unmute_member(self):
        muted = await handle_mute_member_role_based(
            {"server_id": "1", "user_id": "10", "mute_role_id": "2"}, self.deps
        )
        self.assertIn("Muted", muted[0].text)

        unmuted = await handle_unmute_member_role_based(
            {"server_id": "1", "user_id": "10", "mute_role_id": "2"}, self.deps
        )
        self.assertIn("Unmuted", unmuted[0].text)

    async def test_permission_drift_check(self):
        result = await handle_permission_drift_check(
            {
                "server_id": "1",
                "baseline_snapshot": {
                    "roles": [
                        {"role_id": "1", "permissions": "8"},
                        {"role_id": "2", "permissions": "4"},
                    ]
                },
            },
            self.deps,
        )
        payload = json.loads(result[0].text)
        self.assertEqual(payload["driftCount"], 1)


if __name__ == "__main__":
    unittest.main()
