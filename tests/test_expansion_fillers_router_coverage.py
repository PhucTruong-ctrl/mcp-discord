import os
import sys
import json
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("DISCORD_TOKEN", "test-token")
os.environ.setdefault("DISCORD_MCP_CONFIRM_SECRET", "test-secret")

from discord_mcp.tools.handlers.router import TOOL_ROUTER, dispatch_tool_call
from discord_mcp.tools.schemas.expansion_fillers import EXPANSION_FILLER_TOOLS


class TestExpansionFillersRouterCoverage(unittest.IsolatedAsyncioTestCase):
    def test_all_expansion_filler_tools_are_routable(self):
        schema_names = {tool.name for tool in EXPANSION_FILLER_TOOLS}
        missing = sorted(name for name in schema_names if name not in TOOL_ROUTER)
        self.assertEqual(missing, [])

    async def test_dispatch_all_expansion_filler_tools(self):
        cases = {
            "remove_member_timeout": {"server_id": "1", "member_id": "2"},
            "unban_member": {"server_id": "1", "member_id": "2", "reason": "appeal"},
            "bulk_ban_members": {
                "server_id": "1",
                "member_ids": ["2", "3"],
                "dry_run": True,
            },
            "prune_inactive_members": {"server_id": "1", "days": 30, "dry_run": True},
            "create_category": {"server_id": "1", "name": "Ops"},
            "rename_category": {"category_id": "10", "name": "Ops 2"},
            "move_category": {"category_id": "10", "position": 1},
            "delete_category": {"category_id": "10", "dry_run": True},
            "create_incident_room": {
                "server_id": "1",
                "name": "incident-001",
                "reason": "outage",
            },
            "append_incident_event": {
                "incident_channel_id": "20",
                "event_text": "Investigating",
                "severity": "high",
            },
            "close_incident": {
                "incident_channel_id": "20",
                "summary": "Resolved",
                "reason": "stabilized",
            },
            "list_auto_moderation_rules": {"server_id": "1"},
            "create_auto_moderation_rule": {
                "server_id": "1",
                "rule": {"name": "spam"},
            },
            "update_auto_moderation_rule": {
                "server_id": "1",
                "rule_id": "r1",
                "rule": {"name": "spam-v2"},
            },
            "automod_export_rules": {"server_id": "1"},
        }

        for name, arguments in cases.items():
            result = await dispatch_tool_call(name, arguments, {"gateway": object()})
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].type, "text")

    async def test_dry_run_destructive_fillers_include_confirm_token(self):
        cases = {
            "bulk_ban_members": {
                "server_id": "1",
                "member_ids": ["2", "3"],
                "dry_run": True,
            },
            "prune_inactive_members": {"server_id": "1", "days": 30, "dry_run": True},
            "delete_category": {"category_id": "10", "dry_run": True},
        }

        for name, arguments in cases.items():
            with self.subTest(tool=name):
                result = await dispatch_tool_call(
                    name, arguments, {"gateway": object()}
                )
                payload = json.loads(result[0].text)
                self.assertEqual(payload["status"], "dry_run")
                self.assertTrue(payload["confirmToken"])


if __name__ == "__main__":
    unittest.main()
