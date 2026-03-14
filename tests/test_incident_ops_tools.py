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

from discord_mcp.tools.handlers.incident_ops import (
    handle_incident_apply_lockdown,
    handle_incident_get_channel_state,
    handle_incident_rollback_lockdown,
    handle_incident_set_channel_state,
)
from discord_mcp.tools.handlers.router import TOOL_ROUTER
from discord_mcp.tools.schemas import compose_tool_registry


def _payload(result):
    return json.loads(result[0].text)


class IncidentOpsToolTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_includes_incident_ops_tools(self):
        names = [tool.name for tool in compose_tool_registry()]
        self.assertIn("incident_get_channel_state", names)
        self.assertIn("incident_set_channel_state", names)
        self.assertIn("incident_apply_lockdown", names)
        self.assertIn("incident_rollback_lockdown", names)

    def test_router_has_incident_aliases(self):
        self.assertIs(
            TOOL_ROUTER["incident_apply_lockdown"],
            TOOL_ROUTER["incident-apply-lockdown"],
        )
        self.assertIs(
            TOOL_ROUTER["incident_rollback_lockdown"],
            TOOL_ROUTER["incident-rollback-lockdown"],
        )

    async def test_channel_state_model_roundtrip(self):
        result = await handle_incident_set_channel_state(
            {
                "channel_id": "123",
                "state": {
                    "send_messages": False,
                    "add_reactions": False,
                    "slowmode_seconds": 10,
                },
            },
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["channel_id"], "123")
        self.assertEqual(payload["state"]["slowmode_seconds"], 10)

        read_result = await handle_incident_get_channel_state(
            {
                "channel_id": "123",
                "state": payload["state"],
            },
            {},
        )
        read_payload = _payload(read_result)
        self.assertEqual(read_payload["state"]["send_messages"], False)
        self.assertEqual(read_payload["state"]["add_reactions"], False)

    async def test_lockdown_dry_run_emits_confirm_token(self):
        result = await handle_incident_apply_lockdown(
            {
                "channel_ids": ["10", "20"],
                "reason": "raid",
                "dry_run": True,
            },
            {},
        )
        payload = _payload(result)
        self.assertEqual(payload["status"], "dry_run")
        self.assertTrue(payload["confirmToken"])

    async def test_lockdown_execute_requires_valid_confirm_token(self):
        dry_run = await handle_incident_apply_lockdown(
            {
                "channel_ids": ["10"],
                "reason": "incident",
                "dry_run": True,
            },
            {},
        )
        token = _payload(dry_run)["confirmToken"]

        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await handle_incident_apply_lockdown(
                {
                    "channel_ids": ["10"],
                    "reason": "incident",
                    "dry_run": False,
                },
                {},
            )

        with self.assertRaisesRegex(ValueError, "Invalid confirm_token"):
            await handle_incident_apply_lockdown(
                {
                    "channel_ids": ["10"],
                    "reason": "incident",
                    "dry_run": False,
                    "confirm_token": "bad",
                },
                {},
            )

        executed = await handle_incident_apply_lockdown(
            {
                "channel_ids": ["10"],
                "reason": "incident",
                "dry_run": False,
                "confirm_token": token,
            },
            {},
        )
        self.assertEqual(_payload(executed)["status"], "applied")

    async def test_rollback_requires_reason_and_confirm_token(self):
        with self.assertRaisesRegex(ValueError, "reason is required"):
            await handle_incident_rollback_lockdown(
                {
                    "channel_ids": ["10"],
                    "dry_run": True,
                },
                {},
            )


if __name__ == "__main__":
    unittest.main()
