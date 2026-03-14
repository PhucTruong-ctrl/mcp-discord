import asyncio
import importlib
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


class FakeGateway:
    def __init__(self):
        self.deleted = []
        self.timeouts = []
        self.kicks = []
        self.bans = []

    async def bulk_delete_messages(
        self, channel_id: str, message_ids, reason: str | None
    ):
        self.deleted.append((channel_id, list(message_ids), reason))
        return len(message_ids)

    async def timeout_member(
        self, server_id: str, member_id: str, minutes: int, reason: str | None
    ):
        self.timeouts.append((server_id, member_id, minutes, reason))

    async def kick_member(self, server_id: str, member_id: str, reason: str | None):
        self.kicks.append((server_id, member_id, reason))

    async def ban_member(
        self, server_id: str, member_id: str, delete_days: int, reason: str | None
    ):
        self.bans.append((server_id, member_id, delete_days, reason))


class ModerationCoreToolsTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_includes_moderation_tools(self):
        schemas = importlib.import_module("discord_mcp.tools.schemas")
        tool_names = [tool.name for tool in schemas.compose_tool_registry()]

        self.assertIn("moderation_bulk_delete", tool_names)
        self.assertIn("moderation_timeout_member", tool_names)
        self.assertIn("moderation_kick_member", tool_names)
        self.assertIn("moderation_ban_member", tool_names)

    def test_router_includes_moderation_handlers(self):
        router = importlib.import_module("discord_mcp.tools.handlers.router")

        self.assertIn("moderation_bulk_delete", router.TOOL_ROUTER)
        self.assertIn("moderation_timeout_member", router.TOOL_ROUTER)
        self.assertIn("moderation_kick_member", router.TOOL_ROUTER)
        self.assertIn("moderation_ban_member", router.TOOL_ROUTER)

    async def test_bulk_delete_dry_run_returns_confirm_token(self):
        moderation = importlib.import_module(
            "discord_mcp.tools.handlers.moderation_core"
        )
        gateway = FakeGateway()

        result = await moderation.handle_moderation_bulk_delete(
            {
                "channel_id": "10",
                "message_ids": ["1", "2"],
                "reason": "cleanup",
                "dry_run": True,
            },
            {"gateway": gateway},
        )

        payload = json.loads(result[0].text)
        self.assertEqual(payload["status"], "dry_run")
        self.assertTrue(payload["confirmToken"])
        self.assertEqual(payload["action"], "moderation_bulk_delete")
        self.assertEqual(gateway.deleted, [])

    async def test_bulk_delete_requires_valid_confirm_token_for_execute(self):
        moderation = importlib.import_module(
            "discord_mcp.tools.handlers.moderation_core"
        )
        gateway = FakeGateway()

        with self.assertRaisesRegex(ValueError, "confirm_token is required"):
            await moderation.handle_moderation_bulk_delete(
                {
                    "channel_id": "10",
                    "message_ids": ["1", "2"],
                    "dry_run": False,
                    "require_confirm": True,
                },
                {"gateway": gateway},
            )

        dry_run = await moderation.handle_moderation_bulk_delete(
            {
                "channel_id": "10",
                "message_ids": ["1", "2"],
                "dry_run": True,
                "require_confirm": True,
            },
            {"gateway": gateway},
        )
        token = json.loads(dry_run[0].text)["confirmToken"]

        with self.assertRaisesRegex(ValueError, "Invalid confirm_token"):
            await moderation.handle_moderation_bulk_delete(
                {
                    "channel_id": "10",
                    "message_ids": ["1", "2"],
                    "dry_run": False,
                    "require_confirm": True,
                    "confirm_token": "bad-token",
                },
                {"gateway": gateway},
            )

        executed = await moderation.handle_moderation_bulk_delete(
            {
                "channel_id": "10",
                "message_ids": ["1", "2"],
                "dry_run": False,
                "require_confirm": True,
                "confirm_token": token,
            },
            {"gateway": gateway},
        )
        execute_payload = json.loads(executed[0].text)
        self.assertEqual(execute_payload["status"], "executed")
        self.assertEqual(gateway.deleted, [("10", ["1", "2"], None)])

    async def test_missing_secret_raises_when_confirm_required(self):
        moderation = importlib.import_module(
            "discord_mcp.tools.handlers.moderation_core"
        )
        gateway = FakeGateway()
        old = os.environ.pop("DISCORD_MCP_CONFIRM_SECRET", None)
        try:
            with self.assertRaisesRegex(ValueError, "DISCORD_MCP_CONFIRM_SECRET"):
                await moderation.handle_moderation_kick_member(
                    {
                        "server_id": "1",
                        "member_id": "99",
                        "dry_run": False,
                        "require_confirm": True,
                        "confirm_token": "whatever",
                    },
                    {"gateway": gateway},
                )
        finally:
            if old is not None:
                os.environ["DISCORD_MCP_CONFIRM_SECRET"] = old


if __name__ == "__main__":
    unittest.main()
