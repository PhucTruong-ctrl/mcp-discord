import json
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
    handle_automod_get_ruleset,
    handle_automod_rollback_ruleset,
    handle_automod_validate_ruleset,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER
from discord_mcp.tools.schemas import compose_tool_registry


def _payload(result):
    return json.loads(result[0].text)


class AutomodPolicyToolTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_includes_automod_policy_tools(self):
        names = [tool.name for tool in compose_tool_registry()]
        self.assertIn("automod_validate_ruleset", names)
        self.assertIn("automod_get_ruleset", names)
        self.assertIn("automod_apply_ruleset", names)
        self.assertIn("automod_rollback_ruleset", names)

    def test_router_has_automod_aliases(self):
        self.assertIs(
            TOOL_ROUTER["automod_apply_ruleset"],
            TOOL_ROUTER["automod-apply-ruleset"],
        )
        self.assertIs(
            TOOL_ROUTER["automod_rollback_ruleset"],
            TOOL_ROUTER["automod-rollback-ruleset"],
        )

    async def test_validate_ruleset_accepts_caller_supplied_rules(self):
        result = await handle_automod_validate_ruleset(
            {
                "ruleset": {
                    "name": "baseline",
                    "rules": [
                        {
                            "name": "block-spam-links",
                            "trigger_type": "keyword",
                            "trigger_metadata": {"keyword_filter": ["discord.gg/"]},
                            "actions": [{"type": "block_message"}],
                            "enabled": True,
                        }
                    ],
                }
            },
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "valid")
        self.assertEqual(payload["ruleset"]["name"], "baseline")

    async def test_get_ruleset_echoes_ruleset_model(self):
        ruleset = {
            "name": "runtime",
            "rules": [{"name": "caps", "trigger_type": "keyword", "actions": []}],
        }
        result = await handle_automod_get_ruleset(
            {"guild_id": "123", "ruleset": ruleset},
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["guild_id"], "123")
        self.assertEqual(payload["ruleset"]["name"], "runtime")

    async def test_apply_and_rollback_require_reason_and_confirm_token(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            await handle_automod_apply_ruleset(
                {
                    "guild_id": "1",
                    "ruleset": {"name": "baseline", "rules": []},
                    "dry_run": True,
                },
                {},
            )

        dry_run = await handle_automod_apply_ruleset(
            {
                "guild_id": "1",
                "ruleset": {"name": "baseline", "rules": []},
                "reason": "incident",
                "dry_run": True,
            },
            {},
        )
        token = _payload(dry_run)["confirm_token"]

        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await handle_automod_apply_ruleset(
                {
                    "guild_id": "1",
                    "ruleset": {"name": "baseline", "rules": []},
                    "reason": "incident",
                    "dry_run": False,
                },
                {},
            )

        applied = await handle_automod_apply_ruleset(
            {
                "guild_id": "1",
                "ruleset": {"name": "baseline", "rules": []},
                "reason": "incident",
                "dry_run": False,
                "confirm_token": token,
            },
            {},
        )
        self.assertEqual(_payload(applied)["status"], "applied")

        rollback_dry_run = await handle_automod_rollback_ruleset(
            {
                "guild_id": "1",
                "ruleset_name": "baseline",
                "reason": "revert",
                "dry_run": True,
            },
            {},
        )
        rollback_token = _payload(rollback_dry_run)["confirm_token"]
        rolled_back = await handle_automod_rollback_ruleset(
            {
                "guild_id": "1",
                "ruleset_name": "baseline",
                "reason": "revert",
                "dry_run": False,
                "confirm_token": rollback_token,
            },
            {},
        )
        self.assertEqual(_payload(rolled_back)["status"], "rolled_back")


if __name__ == "__main__":
    unittest.main()
