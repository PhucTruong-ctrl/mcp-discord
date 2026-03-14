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
    handle_prune_inactive_members,
)


class TestConfirmTokenEnforcementMatrix(unittest.IsolatedAsyncioTestCase):
    async def test_confirm_token_required_for_all_10_execute_paths(self):
        dummy_deps = {"gateway": object()}
        cases = [
            (
                handle_moderation_bulk_delete,
                {
                    "channel_id": "10",
                    "message_ids": ["1", "2"],
                    "reason": "cleanup",
                    "dry_run": False,
                    "require_confirm": True,
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
                    "require_confirm": True,
                },
            ),
            (
                handle_moderation_kick_member,
                {
                    "server_id": "1",
                    "member_id": "2",
                    "reason": "kick",
                    "dry_run": False,
                    "require_confirm": True,
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
                    "require_confirm": True,
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
        ]

        for handler, arguments in cases:
            with self.assertRaisesRegex(ValueError, "confirm_token is required"):
                await handler(arguments, dummy_deps)


if __name__ == "__main__":
    unittest.main()
