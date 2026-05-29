"""
Runtime contract tests for expansion filler and adjacent tool families.

These tests encode the CURRENT expected behavior of tool families that return
synthetic/placeholder responses (no real Discord API calls). If a handler is
upgraded to make live Discord calls, these tests MUST be updated to match the
new real-behavior contract — they serve as a regression guard against silent
behavior drift.

The following tool families are covered here:

  - Expansion fillers (15 tools, indices 92-106 in canonical registry):
    bulk_ban_members, prune_inactive_members, remove_member_timeout,
    unban_member, create_category, rename_category, move_category,
    delete_category, create_incident_room, append_incident_event,
    close_incident, list_auto_moderation_rules, create_auto_moderation_rule,
    update_auto_moderation_rule, automod_export_rules

  - Incident operations (4 tools):
    incident_get_channel_state, incident_set_channel_state,
    incident_apply_lockdown, incident_rollback_lockdown

  - AutoMod policy tools (4 tools):
    automod_validate_ruleset, automod_get_ruleset,
    automod_apply_ruleset, automod_rollback_ruleset
"""

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

from discord_mcp.tools.handlers.expansion_fillers import (
    handle_append_incident_event,
    handle_automod_export_rules,
    handle_bulk_ban_members,
    handle_close_incident,
    handle_create_auto_moderation_rule,
    handle_create_category,
    handle_create_incident_room,
    handle_delete_category,
    handle_list_auto_moderation_rules,
    handle_move_category,
    handle_prune_inactive_members,
    handle_remove_member_timeout,
    handle_rename_category,
    handle_unban_member,
    handle_update_auto_moderation_rule,
)
from discord_mcp.tools.handlers.incident_ops import (
    handle_incident_apply_lockdown,
    handle_incident_get_channel_state,
    handle_incident_rollback_lockdown,
    handle_incident_set_channel_state,
)
from discord_mcp.tools.handlers.automod_policy import (
    handle_automod_apply_ruleset,
    handle_automod_get_ruleset,
    handle_automod_rollback_ruleset,
    handle_automod_validate_ruleset,
)
from discord_mcp.tools.schemas import compose_tool_registry
from discord_mcp.tools.schemas.expansion_fillers import EXPANSION_FILLER_TOOLS


def _payload(result):
    return json.loads(result[0].text)


class ExpansionFillerContractTests(unittest.IsolatedAsyncioTestCase):
    """Contract: all 15 expansion filler handlers return synthetic responses.
    They do NOT call deps['gateway'] or make Discord API calls.
    """

    def test_expansion_filler_tools_are_in_registry(self):
        names = {tool.name for tool in compose_tool_registry()}
        filler_names = {tool.name for tool in EXPANSION_FILLER_TOOLS}
        self.assertTrue(filler_names.issubset(names))
        self.assertEqual(len(filler_names), 15)

    async def test_remove_member_timeout_returns_synthetic_applied(self):
        result = await handle_remove_member_timeout(
            {"server_id": "1", "member_id": "2"}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["action"], "remove_member_timeout")

    async def test_unban_member_returns_synthetic_applied(self):
        result = await handle_unban_member(
            {"server_id": "1", "member_id": "2", "reason": "appeal"}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["action"], "unban_member")

    async def test_bulk_ban_members_dry_run_returns_confirm_token(self):
        result = await handle_bulk_ban_members(
            {"server_id": "1", "member_ids": ["2", "3"], "dry_run": True}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "dry_run")
        self.assertIn("confirmToken", payload)

    async def test_bulk_ban_members_non_dry_run_requires_confirm_token(self):
        with self.assertRaises(ValueError):
            await handle_bulk_ban_members(
                {"server_id": "1", "member_ids": ["2", "3"], "dry_run": False}, {}
            )

    async def test_prune_inactive_members_dry_run_returns_confirm_token(self):
        result = await handle_prune_inactive_members(
            {"server_id": "1", "days": 30, "dry_run": True}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "dry_run")
        self.assertIn("confirmToken", payload)

    async def test_create_category_returns_synthetic_applied(self):
        result = await handle_create_category({"server_id": "1", "name": "Ops"}, {})
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["action"], "create_category")

    async def test_rename_category_returns_synthetic_applied(self):
        result = await handle_rename_category(
            {"category_id": "10", "name": "Ops 2"}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["action"], "rename_category")

    async def test_move_category_returns_synthetic_applied(self):
        result = await handle_move_category({"category_id": "10", "position": 1}, {})
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["action"], "move_category")

    async def test_delete_category_dry_run_returns_confirm_token(self):
        result = await handle_delete_category(
            {"category_id": "10", "dry_run": True}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "dry_run")
        self.assertIn("confirmToken", payload)

    async def test_create_incident_room_returns_synthetic_applied(self):
        result = await handle_create_incident_room(
            {"server_id": "1", "name": "inc-001", "reason": "outage"}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["action"], "create_incident_room")

    async def test_append_incident_event_returns_synthetic_applied(self):
        result = await handle_append_incident_event(
            {
                "incident_channel_id": "20",
                "event_text": "Investigating",
                "severity": "high",
            },
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["action"], "append_incident_event")

    async def test_close_incident_returns_synthetic_applied(self):
        result = await handle_close_incident(
            {
                "incident_channel_id": "20",
                "summary": "Resolved",
                "reason": "stabilized",
            },
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["action"], "close_incident")

    async def test_list_auto_moderation_rules_returns_empty_rules(self):
        result = await handle_list_auto_moderation_rules({"server_id": "1"}, {})
        payload = _payload(result)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["rules"], [])

    async def test_create_auto_moderation_rule_returns_synthetic_applied(self):
        result = await handle_create_auto_moderation_rule(
            {"server_id": "1", "rule": {"name": "spam"}}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["action"], "create_auto_moderation_rule")

    async def test_update_auto_moderation_rule_returns_synthetic_applied(self):
        result = await handle_update_auto_moderation_rule(
            {"server_id": "1", "rule_id": "r1", "rule": {"name": "spam-v2"}}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "applied")
        self.assertEqual(payload["action"], "update_auto_moderation_rule")

    async def test_automod_export_rules_returns_empty_export(self):
        result = await handle_automod_export_rules({"server_id": "1"}, {})
        payload = _payload(result)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["export"], {"rules": []})

    async def test_never_uses_deps_gateway(self):
        """All expansion fillers must complete successfully with empty deps."""
        harmless = [
            (handle_remove_member_timeout, {"server_id": "1", "member_id": "2"}),
            (handle_unban_member, {"server_id": "1", "member_id": "2"}),
            (handle_create_category, {"server_id": "1", "name": "X"}),
            (handle_rename_category, {"category_id": "10", "name": "X"}),
            (handle_move_category, {"category_id": "10", "position": 1}),
            (
                handle_create_incident_room,
                {"server_id": "1", "name": "X", "reason": "R"},
            ),
            (
                handle_append_incident_event,
                {"incident_channel_id": "20", "event_text": "T", "severity": "low"},
            ),
            (
                handle_close_incident,
                {"incident_channel_id": "20", "summary": "S", "reason": "R"},
            ),
            (handle_list_auto_moderation_rules, {"server_id": "1"}),
            (
                handle_create_auto_moderation_rule,
                {"server_id": "1", "rule": {"name": "X"}},
            ),
            (
                handle_update_auto_moderation_rule,
                {"server_id": "1", "rule_id": "R", "rule": {"name": "X"}},
            ),
            (handle_automod_export_rules, {"server_id": "1"}),
        ]
        for handler, arguments in harmless:
            with self.subTest(handler=handler.__name__):
                result = await handler(arguments, {})
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].type, "text")


class IncidentOpsContractTests(unittest.IsolatedAsyncioTestCase):
    """Contract: incident ops handlers return synthetic responses.
    They use dry_run/confirm_token for destructive actions but do NOT
    make Discord API calls.
    """

    async def test_get_channel_state_echoes_state(self):
        result = await handle_incident_get_channel_state(
            {"channel_id": "10", "state": {"locked": True}}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["channel_id"], "10")
        self.assertTrue(payload["state"]["locked"])

    async def test_set_channel_state_echoes_state(self):
        result = await handle_incident_set_channel_state(
            {"channel_id": "10", "state": {"locked": True}}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["channel_id"], "10")
        self.assertTrue(payload["state"]["locked"])

    async def test_apply_lockdown_dry_run_returns_confirm_token(self):
        result = await handle_incident_apply_lockdown(
            {"channel_ids": ["10", "11"], "reason": "breach", "dry_run": True}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "dry_run")
        self.assertIn("confirmToken", payload)

    async def test_apply_lockdown_non_dry_run_requires_confirm_token(self):
        with self.assertRaises(ValueError):
            await handle_incident_apply_lockdown(
                {"channel_ids": ["10"], "reason": "breach", "dry_run": False}, {}
            )

    async def test_rollback_lockdown_dry_run_returns_confirm_token(self):
        result = await handle_incident_rollback_lockdown(
            {"channel_ids": ["10"], "reason": "resolved", "dry_run": True}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "dry_run")
        self.assertIn("confirmToken", payload)

    async def test_rollback_lockdown_non_dry_run_requires_confirm_token(self):
        with self.assertRaises(ValueError):
            await handle_incident_rollback_lockdown(
                {"channel_ids": ["10"], "reason": "resolved", "dry_run": False}, {}
            )

    async def test_never_uses_deps_gateway(self):
        """All incident ops handlers must complete with empty deps."""
        result = await handle_incident_get_channel_state(
            {"channel_id": "10", "state": {}}, {}
        )
        self.assertEqual(len(result), 1)


class AutomodPolicyContractTests(unittest.IsolatedAsyncioTestCase):
    """Contract: AutoMod policy tools validate ruleset shape and use
    dry_run/confirm_token for apply/rollback, but do NOT make Discord
    API calls (gateway-independent).
    """

    async def test_validate_ruleset_rejects_missing_name(self):
        with self.assertRaises(ValueError):
            await handle_automod_validate_ruleset({"ruleset": {"rules": []}}, {})

    async def test_validate_ruleset_accepts_valid_ruleset(self):
        result = await handle_automod_validate_ruleset(
            {"ruleset": {"name": "baseline", "rules": []}}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "valid")

    async def test_get_ruleset_echoes_ruleset(self):
        result = await handle_automod_get_ruleset(
            {"guild_id": "123", "ruleset": {"name": "baseline", "rules": []}}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["guild_id"], "123")
        self.assertEqual(payload["ruleset"]["name"], "baseline")

    async def test_apply_ruleset_dry_run_returns_confirm_token(self):
        result = await handle_automod_apply_ruleset(
            {
                "guild_id": "1",
                "ruleset": {"name": "baseline", "rules": []},
                "reason": "incident",
                "dry_run": True,
            },
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "dry_run")
        self.assertIn("confirmToken", payload)

    async def test_rollback_ruleset_dry_run_returns_confirm_token(self):
        result = await handle_automod_rollback_ruleset(
            {
                "guild_id": "1",
                "ruleset_name": "baseline",
                "reason": "revert",
                "dry_run": True,
            },
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "dry_run")
        self.assertIn("confirmToken", payload)

    async def test_never_uses_deps_gateway(self):
        """All automod policy handlers must complete with empty deps."""
        result = await handle_automod_validate_ruleset(
            {"ruleset": {"name": "test", "rules": []}}, {}
        )
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
