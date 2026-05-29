import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock


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


def _mock_automod_rule(name="test-rule", rule_id="111", **overrides):
    """Build a mock object with AutoModRule-like interface."""
    rule = MagicMock()
    rule.id = int(rule_id)
    rule.name = name
    rule.guild_id = 123
    rule.event_type = "AutoModRuleEventType.message_send"
    rule.trigger_type = "AutoModRuleTriggerType.keyword"
    rule.trigger_metadata = {}
    rule.enabled = overrides.get("enabled", True)
    rule.exempt_roles = []
    rule.exempt_channels = []
    rule.creator_id = 456
    rule.created_at = None

    action = MagicMock()
    action.type = "AutoModRuleActionType.block_message"
    action.custom_message = "Blocked"
    action.channel_id = None
    action.duration = None
    rule.actions = [action]

    for k, v in overrides.items():
        if k != "enabled":
            setattr(rule, k, v)
    return rule


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

    # --- automod_get_ruleset: now fetches from Discord API ---

    async def test_get_ruleset_fetches_from_api(self):
        """get_ruleset should fetch rules from Discord, not echo caller input."""
        rule = _mock_automod_rule()
        guild = AsyncMock()
        guild.id = 123
        guild.fetch_automod_rules = AsyncMock(return_value=[rule])
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        result = await handle_automod_get_ruleset(
            {"guild_id": "123"},
            deps,
        )
        payload = _payload(result)

        self.assertEqual(payload["guild_id"], "123")
        self.assertEqual(len(payload["rules"]), 1)
        self.assertEqual(payload["rules"][0]["name"], "test-rule")
        guild.fetch_automod_rules.assert_awaited_once()

    async def test_get_ruleset_filters_by_ruleset_name(self):
        """get_ruleset should filter rules by name when ruleset_name provided."""
        rule_a = _mock_automod_rule(name="alpha", rule_id="1")
        rule_b = _mock_automod_rule(name="beta", rule_id="2")
        guild = AsyncMock()
        guild.id = 123
        guild.fetch_automod_rules = AsyncMock(return_value=[rule_a, rule_b])
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        result = await handle_automod_get_ruleset(
            {"guild_id": "123", "ruleset_name": "alpha"},
            deps,
        )
        payload = _payload(result)

        self.assertEqual(len(payload["rules"]), 1)
        self.assertEqual(payload["rules"][0]["name"], "alpha")

    async def test_get_ruleset_returns_empty_when_no_rules(self):
        """get_ruleset should return empty list when no rules exist."""
        guild = AsyncMock()
        guild.id = 123
        guild.fetch_automod_rules = AsyncMock(return_value=[])
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        result = await handle_automod_get_ruleset(
            {"guild_id": "123"},
            deps,
        )
        payload = _payload(result)
        self.assertEqual(payload["rules"], [])

    async def test_get_ruleset_gateway_unavailable_returns_unsupported(self):
        """get_ruleset should return unsupported when no gateway available."""
        result = await handle_automod_get_ruleset({"guild_id": "123"}, {})
        payload = _payload(result)
        self.assertEqual(payload["status"], "unsupported")

    # --- automod_apply_ruleset: now creates rules via Discord API ---

    async def test_apply_ruleset_requires_reason(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            await handle_automod_apply_ruleset(
                {
                    "guild_id": "1",
                    "ruleset": {"name": "baseline", "rules": []},
                    "dry_run": True,
                },
                {},
            )

    async def test_apply_ruleset_dry_run_returns_confirm_token(self):
        """apply dry_run should return confirm token without calling API."""
        result = await handle_automod_apply_ruleset(
            {
                "guild_id": "1",
                "ruleset": {"name": "baseline", "rules": []},
                "reason": "incident",
                "dry_run": True,
            },
            {},
        )
        token = _payload(result)["confirmToken"]
        self.assertIsNotNone(token)

    async def test_apply_ruleset_requires_confirm_token_for_execute(self):
        """apply without confirm_token should raise."""
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

    async def test_apply_ruleset_execute_calls_create_automod_rule(self):
        """apply execute path should create rules via Discord API."""
        guild = AsyncMock()
        guild.id = 123
        guild.create_automod_rule = AsyncMock(
            return_value=_mock_automod_rule(name="created-rule", rule_id="444")
        )
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        # First get a confirm token
        dry_run = await handle_automod_apply_ruleset(
            {
                "guild_id": "1",
                "ruleset": {"name": "baseline", "rules": []},
                "reason": "incident",
                "dry_run": True,
            },
            {},
        )
        token = _payload(dry_run)["confirmToken"]

        # Now execute
        result = await handle_automod_apply_ruleset(
            {
                "guild_id": "1",
                "ruleset": {
                    "name": "baseline",
                    "rules": [
                        {
                            "name": "block-spam",
                            "trigger_type": "keyword",
                            "trigger_metadata": {"keyword_filter": ["bad"]},
                            "actions": [{"type": "block_message"}],
                            "enabled": True,
                        }
                    ],
                },
                "reason": "incident",
                "dry_run": False,
                "confirm_token": token,
            },
            deps,
        )
        payload = _payload(result)

        self.assertEqual(payload["status"], "applied")
        guild.create_automod_rule.assert_awaited_once()
        call_kwargs = guild.create_automod_rule.call_args.kwargs
        self.assertEqual(call_kwargs["name"], "block-spam")

    # --- automod_rollback_ruleset: returns explicit unsupported error ---

    async def test_rollback_ruleset_returns_unsupported(self):
        """rollback should return an explicit unsupported error, not pretend."""
        result = await handle_automod_rollback_ruleset(
            {
                "guild_id": "1",
                "ruleset_name": "baseline",
                "reason": "revert",
            },
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "unsupported")
        self.assertIn("Rollback", payload.get("message", ""))

    async def test_apply_ruleset_valid_token_but_no_gateway_returns_unsupported(
        self,
    ):
        """apply execute with valid confirm_token but no gateway should return
        unsupported, not synthetic applied+empty rules."""
        # Get a valid confirm token via dry_run
        dry_run = await handle_automod_apply_ruleset(
            {
                "guild_id": "1",
                "ruleset": {"name": "baseline", "rules": []},
                "reason": "incident",
                "dry_run": True,
            },
            {},
        )
        token = _payload(dry_run)["confirmToken"]

        # Execute with valid token but NO gateway in deps
        result = await handle_automod_apply_ruleset(
            {
                "guild_id": "1",
                "ruleset": {
                    "name": "baseline",
                    "rules": [
                        {
                            "name": "block-spam",
                            "trigger_type": "keyword",
                            "trigger_metadata": {"keyword_filter": ["bad"]},
                            "actions": [{"type": "block_message"}],
                            "enabled": True,
                        }
                    ],
                },
                "reason": "incident",
                "dry_run": False,
                "confirm_token": token,
            },
            {},  # no gateway
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "unsupported")
        self.assertIn("not available", payload.get("message", ""))
        # Must NOT contain synthetic applied+empty fields
        self.assertNotIn("rules", payload)
        self.assertNotIn("applied", payload.get("status", ""))

    async def test_apply_ruleset_partial_failure_reports_success_and_errors(self):
        """apply should report partial success/failure when some rules fail."""
        guild = AsyncMock()
        guild.id = 123

        async def create_rule(**kwargs):
            if kwargs.get("name") == "fail-rule":
                raise Exception("API error creating rule")
            return _mock_automod_rule(name=kwargs.get("name", "rule"), rule_id="444")

        guild.create_automod_rule = AsyncMock(side_effect=create_rule)
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        # Get confirm token first
        dry_run = await handle_automod_apply_ruleset(
            {
                "guild_id": "1",
                "ruleset": {"name": "baseline", "rules": []},
                "reason": "incident",
                "dry_run": True,
            },
            {},
        )
        token = _payload(dry_run)["confirmToken"]

        # Execute with mixed success/failure rules
        result = await handle_automod_apply_ruleset(
            {
                "guild_id": "1",
                "ruleset": {
                    "name": "baseline",
                    "rules": [
                        {
                            "name": "good-rule",
                            "trigger_type": "keyword",
                            "trigger_metadata": {"keyword_filter": ["bad"]},
                            "actions": [{"type": "block_message"}],
                            "enabled": True,
                        },
                        {
                            "name": "fail-rule",
                            "trigger_type": "keyword",
                            "trigger_metadata": {"keyword_filter": ["bad"]},
                            "actions": [{"type": "block_message"}],
                            "enabled": True,
                        },
                        {
                            "name": "another-good",
                            "trigger_type": "keyword",
                            "trigger_metadata": {"keyword_filter": ["bad"]},
                            "actions": [{"type": "block_message"}],
                            "enabled": True,
                        },
                    ],
                },
                "reason": "incident",
                "dry_run": False,
                "confirm_token": token,
            },
            deps,
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied_with_errors")
        self.assertEqual(len(payload["rules"]), 2)
        self.assertEqual(len(payload["errors"]), 1)
        self.assertIn("fail-rule", payload["errors"][0])


if __name__ == "__main__":
    unittest.main()
