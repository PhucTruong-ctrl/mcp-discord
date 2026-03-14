import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

from discord_mcp.tools.handlers.automod_policy import (
    handle_automod_apply_ruleset,
    handle_automod_rollback_ruleset,
)
from discord_mcp.tools.handlers.incident_ops import (
    handle_incident_apply_lockdown,
    handle_incident_rollback_lockdown,
)
from discord_mcp.tools.handlers.moderation_core import (
    handle_moderation_ban_member,
    handle_moderation_bulk_delete,
    handle_moderation_kick_member,
    handle_moderation_timeout_member,
)
from discord_mcp.tools.handlers.expansion_fillers import (
    handle_bulk_ban_members,
    handle_delete_category,
    handle_prune_inactive_members,
)
from discord_mcp.tools.handlers.role_governance import (
    handle_add_roles_bulk,
    handle_remove_roles_bulk,
)


class _FakeRole:
    def __init__(self, role_id: int):
        self.id = role_id


class _FakeMember:
    async def add_roles(self, *roles, reason=None):
        return None

    async def remove_roles(self, *roles, reason=None):
        return None


class _FakeGuild:
    def __init__(self):
        self.roles = [_FakeRole(1), _FakeRole(2)]

    def get_role(self, role_id: int):
        for role in self.roles:
            if role.id == role_id:
                return role
        return None

    async def fetch_member(self, user_id: int):
        return _FakeMember()


class _FakeGateway:
    async def bulk_delete_messages(self, channel_id, message_ids, reason):
        return len(message_ids)

    async def timeout_member(self, server_id, member_id, duration_minutes, reason):
        return None

    async def kick_member(self, server_id, member_id, reason):
        return None

    async def ban_member(self, server_id, member_id, delete_message_days, reason):
        return None

    async def fetch_guild(self, _server_id):
        return _FakeGuild()


class TestConfirmTokenEnforcementMatrix(unittest.IsolatedAsyncioTestCase):
    async def test_confirm_token_required_for_all_13_execute_paths(self):
        dummy_deps = {"gateway": _FakeGateway()}
        cases = [
            (
                handle_moderation_bulk_delete,
                {
                    "channel_id": "10",
                    "message_ids": ["1", "2"],
                    "reason": "cleanup",
                    "dry_run": False,
                },
            ),
            (
                handle_moderation_timeout_member,
                {
                    "server_id": "1",
                    "member_id": "2",
                    "duration_minutes": 30,
                    "reason": "timeout",
                    "dry_run": False,
                },
            ),
            (
                handle_moderation_kick_member,
                {
                    "server_id": "1",
                    "member_id": "2",
                    "reason": "kick",
                    "dry_run": False,
                },
            ),
            (
                handle_moderation_ban_member,
                {
                    "server_id": "1",
                    "member_id": "2",
                    "delete_message_days": 1,
                    "reason": "ban",
                    "dry_run": False,
                },
            ),
            (
                handle_incident_apply_lockdown,
                {
                    "channel_ids": ["10", "11"],
                    "reason": "incident",
                    "dry_run": False,
                },
            ),
            (
                handle_incident_rollback_lockdown,
                {
                    "channel_ids": ["10", "11"],
                    "reason": "rollback",
                    "dry_run": False,
                },
            ),
            (
                handle_automod_apply_ruleset,
                {
                    "guild_id": "100",
                    "ruleset": {"name": "baseline", "rules": []},
                    "reason": "apply",
                    "dry_run": False,
                },
            ),
            (
                handle_automod_rollback_ruleset,
                {
                    "guild_id": "100",
                    "ruleset_name": "baseline",
                    "reason": "rollback",
                    "dry_run": False,
                },
            ),
            (
                handle_bulk_ban_members,
                {
                    "server_id": "100",
                    "member_ids": ["10", "11"],
                    "reason": "bulk-ban",
                    "dry_run": False,
                },
            ),
            (
                handle_prune_inactive_members,
                {
                    "server_id": "100",
                    "days": 30,
                    "reason": "prune",
                    "dry_run": False,
                },
            ),
            (
                handle_add_roles_bulk,
                {
                    "server_id": "1",
                    "user_ids": ["10", "11"],
                    "role_ids": ["1", "2"],
                    "reason": "bulk-add-roles",
                    "dry_run": False,
                },
            ),
            (
                handle_remove_roles_bulk,
                {
                    "server_id": "1",
                    "user_ids": ["10", "11"],
                    "role_ids": ["1", "2"],
                    "reason": "bulk-remove-roles",
                    "dry_run": False,
                },
            ),
            (
                handle_delete_category,
                {
                    "category_id": "123",
                    "reason": "cleanup",
                    "dry_run": False,
                },
            ),
        ]

        for handler, arguments in cases:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "confirm_token is required"):
                    await handler(arguments, dummy_deps)


if __name__ == "__main__":
    unittest.main()
