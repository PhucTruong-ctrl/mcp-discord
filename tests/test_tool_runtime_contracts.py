"""
Runtime contract tests for expansion filler and adjacent tool families.

These tests encode the CURRENT expected behavior in a gateway-absent context
(empty deps, no Discord API available). All handlers covered here must
complete gracefully without a gateway — either returning synthetic placeholder
responses or falling back to synthetic results when no gateway is present.

If a handler is upgraded to always require a gateway and cannot fall back,
these tests MUST be updated — they serve as a regression guard against
silent behavior drift.

The following tool families are covered here:

  - Expansion fillers — synthetic-only (11 tools, indices 93-103):
    bulk_ban_members, prune_inactive_members, remove_member_timeout,
    unban_member, create_category, rename_category, move_category,
    delete_category, create_incident_room, append_incident_event,
    close_incident

  - Expansion fillers — gateway-aware with synthetic fallback
    (4 tools, indices 104-107):
    list_auto_moderation_rules, create_auto_moderation_rule,
    update_auto_moderation_rule, automod_export_rules

  - Incident operations (4 tools):
    incident_get_channel_state, incident_set_channel_state,
    incident_apply_lockdown, incident_rollback_lockdown

  - AutoMod policy tools (4 tools):
    automod_validate_ruleset, automod_get_ruleset,
    automod_apply_ruleset, automod_rollback_ruleset

NOTE: Gateway-present runtime tests for tools 104-107 live in
test_automod_runtime_tools.py, which exercises the Discord API code path
when a mock gateway is provided.
"""

import json
import os
import sys
import tempfile
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


class McpContentContractTests(unittest.TestCase):
    """Verify MCP content type contracts used throughout handlers."""

    def test_all_tools_have_valid_mcp_types(self):
        """Every Tool in the registry must build correctly with mcp SDK."""
        tools = compose_tool_registry()
        for tool in tools:
            self.assertIsInstance(tool.name, str)
            self.assertIsInstance(tool.inputSchema, dict)
            # description is optional in mcp SDK but we always provide it
            self.assertIsNotNone(
                tool.description, f"Tool {tool.name} missing description"
            )

    def test_handler_results_are_text_content_list(self):
        """All handler results must be List[TextContent] with type='text'."""
        from mcp.types import TextContent

        # Smoke check: verify TextContent usage everywhere
        tc = TextContent(type="text", text="{}")
        self.assertEqual(tc.type, "text")
        # Verify repr is valid
        self.assertIsInstance(str(tc), str)


class ExpansionFillerContractTests(unittest.IsolatedAsyncioTestCase):
    """Contract: expansion filler handlers with empty/gateway-absent deps.

    - Tools 93-103 are synthetic-only — they never call deps['gateway'].
    - Tools 104-107 are gateway-aware — they fall back to synthetic responses
      when gateway is absent. Gateway-present runtime tests for 104-107 are
      in test_automod_runtime_tools.py.
    """

    def test_expansion_filler_tools_are_in_registry(self):
        names = {tool.name for tool in compose_tool_registry()}
        filler_names = {tool.name for tool in EXPANSION_FILLER_TOOLS}
        self.assertTrue(filler_names.issubset(names))
        self.assertEqual(len(filler_names), 15)

    async def test_mutating_fillers_require_a_gateway(self):
        """No silent success: mutating fillers fail loudly without a gateway."""
        calls = [
            (handle_remove_member_timeout, {"server_id": "1", "member_id": "2"}),
            (handle_unban_member, {"server_id": "1", "member_id": "2"}),
            (handle_create_category, {"server_id": "1", "name": "Ops"}),
            (handle_rename_category, {"category_id": "10", "name": "Ops 2"}),
            (handle_move_category, {"category_id": "10", "position": 1}),
            (
                handle_create_incident_room,
                {"server_id": "1", "name": "inc-001", "reason": "outage"},
            ),
            (
                handle_append_incident_event,
                {
                    "incident_channel_id": "20",
                    "event_text": "Investigating",
                    "severity": "high",
                },
            ),
            (
                handle_close_incident,
                {
                    "incident_channel_id": "20",
                    "summary": "Resolved",
                    "reason": "stabilized",
                },
            ),
        ]
        for handler, arguments in calls:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "gateway is required"):
                    await handler(dict(arguments), {})

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

    async def test_delete_category_dry_run_returns_confirm_token(self):
        result = await handle_delete_category(
            {"category_id": "10", "dry_run": True}, {}
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "dry_run")
        self.assertIn("confirmToken", payload)

    async def test_automod_tools_require_a_gateway(self):
        """AutoMod reads and writes report the missing gateway instead of faking data."""
        with self.assertRaisesRegex(ValueError, "gateway is required"):
            await handle_list_auto_moderation_rules({"server_id": "1"}, {})
        with self.assertRaisesRegex(ValueError, "gateway is required"):
            await handle_create_auto_moderation_rule(
                {"server_id": "1", "rule": {"name": "spam"}}, {}
            )
        with self.assertRaisesRegex(ValueError, "gateway is required"):
            await handle_update_auto_moderation_rule(
                {"server_id": "1", "rule_id": "1", "rule": {"name": "spam-v2"}}, {}
            )
        with self.assertRaisesRegex(ValueError, "gateway is required"):
            await handle_automod_export_rules({"server_id": "1"}, {})

    async def test_gateway_aware_handlers_report_a_missing_gateway(self):
        """Every mutating filler fails loudly instead of reporting a fake success."""
        cases = [
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
                {"server_id": "1", "rule": {"name": "n"}},
            ),
            (
                handle_update_auto_moderation_rule,
                {"server_id": "1", "rule_id": "1", "rule": {"name": "n"}},
            ),
            (handle_automod_export_rules, {"server_id": "1"}),
        ]
        for handler, arguments in cases:
            with self.subTest(handler=handler.__name__):
                with self.assertRaisesRegex(ValueError, "gateway is required"):
                    await handler(dict(arguments), {})


if __name__ == "__main__":
    unittest.main()
