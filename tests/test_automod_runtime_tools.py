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

from discord_mcp.tools.handlers.expansion_fillers import (
    handle_automod_export_rules,
    handle_create_auto_moderation_rule,
    handle_list_auto_moderation_rules,
    handle_update_auto_moderation_rule,
)


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


class AutomodRuntimeToolTests(unittest.IsolatedAsyncioTestCase):
    """Test that runtime AutoMod tools call Discord API instead of returning placeholders.

    These tools (104-107 in the canonical registry) are *gateway-aware*: when
    a gateway is available they call the Discord API; when it is absent they
    fall back to synthetic placeholder responses. The gateway-absent fallback
    contract is tested in test_tool_runtime_contracts.py (class
    ExpansionFillerContractTests).
    """

    def _make_deps(self, rules=None):
        """Build deps dict with a mock gateway returning given rules."""
        guild = AsyncMock()
        guild.id = 123
        guild.fetch_automod_rules = AsyncMock(return_value=rules or [])
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        return {"gateway": gateway}

    # --- list_auto_moderation_rules ---

    async def test_list_rules_returns_empty_when_no_rules(self):
        """list should return empty list, not placeholder."""
        deps = self._make_deps([])
        result = await handle_list_auto_moderation_rules({"server_id": "123"}, deps)
        payload = _payload(result)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["rules"], [])

    async def test_list_rules_fetches_from_api(self):
        """list should fetch rules via guild.fetch_automod_rules()."""
        rule = _mock_automod_rule()
        deps = self._make_deps([rule])

        result = await handle_list_auto_moderation_rules({"server_id": "123"}, deps)
        payload = _payload(result)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["rules"]), 1)
        self.assertEqual(payload["rules"][0]["name"], "test-rule")
        self.assertEqual(payload["rules"][0]["id"], "111")
        deps["gateway"].resolve_guild.assert_awaited_once()
        deps[
            "gateway"
        ].resolve_guild.return_value.fetch_automod_rules.assert_awaited_once()

    async def test_list_rules_serializes_actions(self):
        """list should serialize automod rule actions."""
        rule = _mock_automod_rule()
        deps = self._make_deps([rule])

        result = await handle_list_auto_moderation_rules({"server_id": "123"}, deps)
        payload = _payload(result)

        actions = payload["rules"][0]["actions"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["type"], "AutoModRuleActionType.block_message")
        self.assertEqual(actions[0]["custom_message"], "Blocked")

    async def test_list_rules_serializes_enabled_flag(self):
        """list should include the enabled flag."""
        rule = _mock_automod_rule(enabled=False)
        deps = self._make_deps([rule])

        result = await handle_list_auto_moderation_rules({"server_id": "123"}, deps)
        payload = _payload(result)

        self.assertFalse(payload["rules"][0]["enabled"])

    async def test_list_rules_gateway_unavailable_returns_empty_synthetic(self):
        """list should return empty synthetic when no gateway available."""
        result = await handle_list_auto_moderation_rules({"server_id": "123"}, {})
        payload = _payload(result)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["rules"], [])

    # --- create_auto_moderation_rule ---

    async def test_create_rule_calls_guild_api(self):
        """create should call guild.create_automod_rule()."""
        guild = AsyncMock()
        guild.id = 123
        new_rule = _mock_automod_rule(name="new-rule", rule_id="222")
        guild.create_automod_rule = AsyncMock(return_value=new_rule)
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        result = await handle_create_auto_moderation_rule(
            {
                "server_id": "123",
                "rule": {
                    "name": "new-rule",
                    "trigger_type": "keyword",
                    "trigger_metadata": {"keyword_filter": ["bad"]},
                    "actions": [{"type": "block_message"}],
                    "enabled": True,
                },
            },
            deps,
        )
        payload = _payload(result)

        self.assertEqual(payload["status"], "applied")
        guild.create_automod_rule.assert_awaited_once()
        call_kwargs = guild.create_automod_rule.call_args.kwargs
        self.assertEqual(call_kwargs["name"], "new-rule")

    async def test_create_rule_returns_serialized_rule(self):
        """create should return the serialized created rule."""
        guild = AsyncMock()
        guild.id = 123
        new_rule = _mock_automod_rule(name="created-rule", rule_id="333")
        guild.create_automod_rule = AsyncMock(return_value=new_rule)
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        result = await handle_create_auto_moderation_rule(
            {
                "server_id": "123",
                "rule": {
                    "name": "created-rule",
                    "trigger_type": "keyword",
                    "actions": [{"type": "block_message"}],
                },
            },
            deps,
        )
        payload = _payload(result)

        self.assertIn("rule", payload)
        self.assertEqual(payload["rule"]["name"], "created-rule")
        self.assertEqual(payload["rule"]["id"], "333")

    async def test_create_rule_gateway_unavailable_returns_synthetic_applied(self):
        """create should return synthetic applied when no gateway available."""
        result = await handle_create_auto_moderation_rule(
            {
                "server_id": "123",
                "rule": {"name": "test", "trigger_type": "keyword", "actions": []},
            },
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")

    async def test_create_rule_passes_reason_to_api(self):
        """create should pass reason to guild.create_automod_rule()."""
        guild = AsyncMock()
        guild.id = 123
        guild.create_automod_rule = AsyncMock(
            return_value=_mock_automod_rule(name="reasoned-rule", rule_id="555")
        )
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        result = await handle_create_auto_moderation_rule(
            {
                "server_id": "123",
                "rule": {
                    "name": "reasoned-rule",
                    "trigger_type": "keyword",
                    "trigger_metadata": {"keyword_filter": ["bad"]},
                    "actions": [{"type": "block_message"}],
                    "enabled": True,
                },
                "reason": "audit-reason-123",
            },
            deps,
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        call_kwargs = guild.create_automod_rule.call_args.kwargs
        self.assertEqual(call_kwargs["reason"], "audit-reason-123")

    # --- update_auto_moderation_rule ---

    async def test_update_rule_calls_edit(self):
        """update should find the rule by ID and call edit()."""
        rule_mock = AsyncMock()
        rule_mock.id = 111
        guild = AsyncMock()
        guild.id = 123
        guild.fetch_automod_rules = AsyncMock(return_value=[rule_mock])
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        result = await handle_update_auto_moderation_rule(
            {
                "server_id": "123",
                "rule_id": "111",
                "rule": {"name": "updated-name", "enabled": False},
            },
            deps,
        )
        payload = _payload(result)

        self.assertEqual(payload["status"], "applied")
        rule_mock.edit.assert_awaited_once_with(name="updated-name", enabled=False)

    async def test_update_rule_raises_on_missing_rule(self):
        """update should raise if rule ID not found."""
        guild = AsyncMock()
        guild.id = 123
        guild.fetch_automod_rules = AsyncMock(return_value=[])
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        with self.assertRaises(ValueError):
            await handle_update_auto_moderation_rule(
                {
                    "server_id": "123",
                    "rule_id": "999",
                    "rule": {"name": "nonexistent"},
                },
                deps,
            )

    async def test_update_rule_gateway_unavailable_returns_synthetic_applied(self):
        """update should return synthetic applied when no gateway available."""
        result = await handle_update_auto_moderation_rule(
            {
                "server_id": "123",
                "rule_id": "111",
                "rule": {"name": "test-rename"},
            },
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")

    async def test_update_rule_passes_reason_to_edit(self):
        """update should pass reason to rule.edit()."""
        rule_mock = AsyncMock()
        rule_mock.id = 111
        guild = AsyncMock()
        guild.id = 123
        guild.fetch_automod_rules = AsyncMock(return_value=[rule_mock])
        gateway = AsyncMock()
        gateway.resolve_guild = AsyncMock(return_value=guild)
        deps = {"gateway": gateway}

        result = await handle_update_auto_moderation_rule(
            {
                "server_id": "123",
                "rule_id": "111",
                "rule": {"name": "updated-name", "enabled": False},
                "reason": "audit-reason-update",
            },
            deps,
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        rule_mock.edit.assert_awaited_once_with(
            name="updated-name", enabled=False, reason="audit-reason-update"
        )

    # --- automod_export_rules ---

    async def test_export_rules_returns_real_rules(self):
        """export should fetch rules from API, not return empty export."""
        rule = _mock_automod_rule(name="exported-rule")
        deps = self._make_deps([rule])

        result = await handle_automod_export_rules({"server_id": "123"}, deps)
        payload = _payload(result)

        self.assertEqual(payload["status"], "ok")
        self.assertIn("export", payload)
        self.assertEqual(len(payload["export"]["rules"]), 1)
        self.assertEqual(payload["export"]["rules"][0]["name"], "exported-rule")
        deps[
            "gateway"
        ].resolve_guild.return_value.fetch_automod_rules.assert_awaited_once()

    async def test_export_rules_empty_when_no_rules(self):
        """export should return empty rules list."""
        deps = self._make_deps([])

        result = await handle_automod_export_rules({"server_id": "123"}, deps)
        payload = _payload(result)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["export"]["rules"], [])

    async def test_export_rules_gateway_unavailable_returns_empty_synthetic(self):
        """export should return empty synthetic when no gateway available."""
        result = await handle_automod_export_rules({"server_id": "123"}, {})
        payload = _payload(result)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["export"]["rules"], [])


if __name__ == "__main__":
    unittest.main()
